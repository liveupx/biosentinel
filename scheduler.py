"""
BioSentinel — Background Scheduler
====================================
Runs periodic background jobs without requiring Celery or Redis.
Uses APScheduler (lightweight, in-process). Jobs survive server restarts
because they re-scan the DB on every run rather than storing job state.

Jobs
----
1. overdue_checkup_scan  — runs every 24 hours at 08:00
   Scans all patients, finds those >90 days since last checkup,
   and sends email / SMS reminders using the existing notify engine.
   Respects per-user email settings and per-patient notify preferences.

2. daily_stats_log  — runs every 24 hours at 06:00
   Logs a brief daily summary (patient count, overdue count, alert count)
   for operational monitoring.

Usage (called from app.py startup)
------------------------------------
  from scheduler import start_scheduler
  start_scheduler(SessionLocal, notify_engine, email_config_engine)

The scheduler runs in a daemon thread — it exits when the main process exits.
"""

import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("biosentinel.scheduler")


def start_scheduler(SessionLocal, notify_engine, email_config_engine):
    """
    Start the APScheduler background scheduler.
    Returns the scheduler instance (or None if APScheduler not installed).
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "APScheduler not installed — background jobs disabled. "
            "Install with: pip install apscheduler"
        )
        return None

    scheduler = BackgroundScheduler(daemon=True, timezone="UTC")

    # ── Job 1: Overdue checkup scan — runs daily at 08:00 UTC ─────────────────
    @scheduler.scheduled_job(CronTrigger(hour=8, minute=0))
    def overdue_checkup_scan():
        _run_overdue_scan(SessionLocal, notify_engine, email_config_engine)

    # ── Job 2: Daily stats log — runs daily at 06:00 UTC ─────────────────────
    @scheduler.scheduled_job(CronTrigger(hour=6, minute=0))
    def daily_stats_log():
        _log_daily_stats(SessionLocal)

    # ── Job 3: Hourly health check log ──────────────────────────────────────
    @scheduler.scheduled_job("interval", hours=1)
    def hourly_ping():
        logger.info("BioSentinel scheduler heartbeat — background jobs running.")

    scheduler.start()
    logger.info(
        "Background scheduler started. Jobs: "
        "overdue_scan@08:00UTC, daily_stats@06:00UTC, hourly_ping"
    )
    return scheduler


def _run_overdue_scan(SessionLocal, notify_engine, email_config_engine):
    """
    Scan all patients and send reminders to those overdue for a checkup.
    Called automatically at 08:00 UTC daily by the scheduler.
    """
    logger.info("Starting scheduled overdue checkup scan...")

    # Import here to avoid circular imports with app.py
    try:
        from app import DBPatient, DBCheckup, DBUser, DBEmailConfig
    except ImportError as e:
        logger.error(f"Scheduler: could not import DB models: {e}")
        return

    db = SessionLocal()
    reminded = 0
    skipped = 0
    errors = 0

    try:
        threshold_days = int(os.getenv("OVERDUE_REMINDER_DAYS", "90"))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=threshold_days)).strftime("%Y-%m-%d")

        patients = db.query(DBPatient).all()

        for pat in patients:
            try:
                # Check notify preference
                if not getattr(pat, "notify_overdue", 1):
                    skipped += 1
                    continue

                # Find last checkup
                last_chk = (
                    db.query(DBCheckup)
                    .filter(DBCheckup.patient_id == pat.id)
                    .order_by(DBCheckup.checkup_date.desc())
                    .first()
                )

                is_overdue = (
                    last_chk is None or
                    last_chk.checkup_date[:10] < cutoff
                )
                if not is_overdue:
                    skipped += 1
                    continue

                # Calculate days overdue
                if last_chk:
                    last_date = datetime.strptime(last_chk.checkup_date[:10], "%Y-%m-%d")
                    days_since = (datetime.now(timezone.utc) - last_date).days
                else:
                    days_since = 999  # no checkup ever

                days_overdue = max(0, days_since - threshold_days)

                # Get owner's email config
                owner = db.query(DBUser).filter(DBUser.id == pat.owner_id).first()
                if not owner:
                    skipped += 1
                    continue

                email_cfg = (
                    db.query(DBEmailConfig)
                    .filter(DBEmailConfig.user_id == owner.id)
                    .first()
                )

                # Send email reminder to the clinician
                if email_cfg and email_config_engine:
                    try:
                        email_config_engine.send_reminder_email(
                            email_cfg,
                            patient_age=pat.age,
                            patient_sex=getattr(pat, "sex", "?"),
                            patient_id=pat.id,
                            days_overdue=days_overdue,
                        )
                        reminded += 1
                        logger.info(
                            f"Reminder sent: patient {pat.id} "
                            f"({days_overdue}d overdue) → {owner.username}"
                        )
                    except Exception as e:
                        logger.warning(f"Reminder email failed for patient {pat.id}: {e}")
                        errors += 1
                else:
                    # Log that reminder was due but no email config
                    logger.info(
                        f"Patient {pat.id} is {days_overdue}d overdue "
                        f"(owner: {owner.username if owner else '?'}) — "
                        f"no email config, skipping"
                    )
                    skipped += 1

            except Exception as e:
                logger.error(f"Overdue scan error for patient {pat.id}: {e}")
                errors += 1

    finally:
        db.close()

    logger.info(
        f"Overdue scan complete: {reminded} reminded, {skipped} skipped, {errors} errors"
    )
    return {"reminded": reminded, "skipped": skipped, "errors": errors}


def _log_daily_stats(SessionLocal):
    """
    Log daily operational stats. Does not send any notifications.
    """
    try:
        from app import DBPatient, DBCheckup, DBAlert, DBPrediction
    except ImportError:
        return

    db = SessionLocal()
    try:
        n_patients   = db.query(DBPatient).count()
        n_checkups   = db.query(DBCheckup).count()
        n_alerts     = db.query(DBAlert).filter(DBAlert.acknowledged == 0).count()
        n_preds      = db.query(DBPrediction).count()

        threshold_days = int(os.getenv("OVERDUE_REMINDER_DAYS", "90"))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=threshold_days)).strftime("%Y-%m-%d")
        n_overdue = 0
        for pat in db.query(DBPatient).all():
            last = (
                db.query(DBCheckup)
                .filter(DBCheckup.patient_id == pat.id)
                .order_by(DBCheckup.checkup_date.desc())
                .first()
            )
            if not last or last.checkup_date[:10] < cutoff:
                n_overdue += 1

        logger.info(
            f"Daily stats — patients:{n_patients} checkups:{n_checkups} "
            f"overdue:{n_overdue} unread_alerts:{n_alerts} predictions:{n_preds}"
        )
    except Exception as e:
        logger.error(f"Daily stats job failed: {e}")
    finally:
        db.close()
