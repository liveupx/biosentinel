#!/usr/bin/env python3
"""
BioSentinel — SQLite → PostgreSQL Migration Script
====================================================
Migrates an existing biosentinel.db (SQLite) to a PostgreSQL database.

Usage
-----
  # 1. Set up PostgreSQL (local or cloud):
  #    createdb biosentinel
  #    createuser biosentinel -P

  # 2. Run this script:
  python migrate_to_postgres.py \\
    --sqlite biosentinel.db \\
    --postgres "postgresql://biosentinel:yourpassword@localhost:5432/biosentinel"

  # 3. Verify:
  python migrate_to_postgres.py --verify \\
    --sqlite biosentinel.db \\
    --postgres "postgresql://biosentinel:yourpassword@localhost:5432/biosentinel"

  # 4. Update your .env:
  #    DATABASE_URL=postgresql://biosentinel:yourpassword@localhost:5432/biosentinel

What it does
------------
- Creates all tables in PostgreSQL (same schema as SQLite)
- Copies every row from SQLite → PostgreSQL in batches
- Preserves all IDs, timestamps, and relationships
- Does NOT delete SQLite — keeps it as backup

Requirements
------------
  pip install sqlalchemy psycopg2-binary
"""

import argparse
import sys
import os
from datetime import datetime

def run_migration(sqlite_url: str, postgres_url: str, batch_size: int = 500):
    print("\n" + "="*60)
    print("  BioSentinel SQLite → PostgreSQL Migration")
    print("="*60 + "\n")

    try:
        from sqlalchemy import create_engine, text, inspect
        from sqlalchemy.orm import sessionmaker
    except ImportError:
        print("❌ SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary")
        sys.exit(1)

    # ── Connect to both databases ──────────────────────────────────────────────
    print(f"📂 Source:      {sqlite_url}")
    print(f"🐘 Destination: {postgres_url[:postgres_url.find('@')+20]}...")
    print()

    sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    pg_engine = create_engine(
        postgres_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

    # ── Test connections ───────────────────────────────────────────────────────
    try:
        with sqlite_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ SQLite connection OK")
    except Exception as e:
        print(f"❌ SQLite connection failed: {e}")
        sys.exit(1)

    try:
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL connection OK")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("   Check your connection string and that PostgreSQL is running.")
        sys.exit(1)

    print()

    # ── Import app models to get the full schema ────────────────────────────
    print("⚙️  Loading BioSentinel schema...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("DATABASE_URL", sqlite_url)
    os.environ.setdefault("SECRET_KEY", "migration-temp-secret")
    os.environ.setdefault("EMAIL_ENABLED", "false")

    try:
        import app as app_module
        Base = app_module.Base
    except Exception as e:
        print(f"❌ Could not import app.py: {e}")
        sys.exit(1)

    # ── Create all tables in PostgreSQL ───────────────────────────────────────
    print("🏗️  Creating tables in PostgreSQL...")
    try:
        Base.metadata.create_all(bind=pg_engine)
        print("✅ All tables created\n")
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        sys.exit(1)

    # ── Get table list in dependency order ────────────────────────────────────
    sqlite_inspector = inspect(sqlite_engine)
    tables = sqlite_inspector.get_table_names()

    # Order matters for foreign keys — parents before children
    ORDERED_TABLES = [
        "users",
        "email_configs",
        "notification_prefs",
        "patients",
        "checkups",
        "predictions",
        "alerts",
        "medications",
        "diagnoses",
        "diet_plans",
        "audit_logs",
        "sessions",
        "password_reset_tokens",
        "genomic_profiles",
        "webhooks",
    ]

    # Add any tables in the DB that aren't in our ordered list
    all_tables = ORDERED_TABLES + [t for t in tables if t not in ORDERED_TABLES]
    # Filter to only tables that actually exist in SQLite
    all_tables = [t for t in all_tables if t in tables]

    # ── Migrate each table ─────────────────────────────────────────────────────
    total_rows = 0
    print("📦 Migrating data...\n")

    with sqlite_engine.connect() as src_conn, pg_engine.connect() as dst_conn:
        for table in all_tables:
            try:
                # Count rows
                count_result = src_conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                row_count = count_result.scalar()

                if row_count == 0:
                    print(f"  ⏭  {table:<30} (empty — skipped)")
                    continue

                print(f"  📋 {table:<30} {row_count:>6} rows", end="", flush=True)

                # Clear destination table (in case of re-run)
                dst_conn.execute(text(f'DELETE FROM "{table}"'))
                dst_conn.commit()

                # Migrate in batches
                offset = 0
                migrated = 0
                while offset < row_count:
                    rows = src_conn.execute(
                        text(f'SELECT * FROM "{table}" LIMIT {batch_size} OFFSET {offset}')
                    ).mappings().all()

                    if not rows:
                        break

                    # Build insert
                    for row in rows:
                        row_dict = dict(row)
                        cols = ", ".join(f'"{k}"' for k in row_dict.keys())
                        placeholders = ", ".join(f":{k}" for k in row_dict.keys())
                        dst_conn.execute(
                            text(f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'),
                            row_dict
                        )

                    dst_conn.commit()
                    migrated += len(rows)
                    offset += batch_size
                    print(f"\r  ✅ {table:<30} {migrated:>6}/{row_count} rows migrated", end="", flush=True)

                total_rows += migrated
                print()  # newline after progress

            except Exception as e:
                print(f"\n  ⚠️  {table}: {e} — skipped")
                dst_conn.rollback()

    print(f"\n✅ Migration complete — {total_rows:,} total rows migrated")
    print(f"   SQLite backup preserved at: {sqlite_url.replace('sqlite:///', '')}")
    print()
    print("📝 Next steps:")
    print("   1. Run --verify to check row counts match")
    print("   2. Update DATABASE_URL in your .env file")
    print("   3. Restart BioSentinel: docker-compose restart")
    print()


def run_verify(sqlite_url: str, postgres_url: str):
    print("\n" + "="*60)
    print("  BioSentinel Migration Verification")
    print("="*60 + "\n")

    from sqlalchemy import create_engine, text, inspect

    sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    pg_engine = create_engine(postgres_url)

    sqlite_inspector = inspect(sqlite_engine)
    tables = sqlite_inspector.get_table_names()

    all_match = True
    print(f"{'Table':<30} {'SQLite':>10} {'PostgreSQL':>12} {'Status':>8}")
    print("-" * 65)

    with sqlite_engine.connect() as src, pg_engine.connect() as dst:
        for table in sorted(tables):
            try:
                sqlite_count = src.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
                pg_count = dst.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
                match = sqlite_count == pg_count
                status = "✅" if match else "❌"
                if not match:
                    all_match = False
                print(f"  {table:<28} {sqlite_count:>10,} {pg_count:>12,}  {status}")
            except Exception as e:
                print(f"  {table:<28} {'error':>10} {'error':>12}  ⚠️  ({e})")
                all_match = False

    print()
    if all_match:
        print("✅ All row counts match — migration verified!")
        print()
        print("📝 Update your .env:")
        print(f"   DATABASE_URL={postgres_url}")
    else:
        print("❌ Row count mismatches found — re-run the migration")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate BioSentinel from SQLite to PostgreSQL"
    )
    parser.add_argument(
        "--sqlite",
        default="sqlite:///./biosentinel.db",
        help="SQLite URL (default: sqlite:///./biosentinel.db)"
    )
    parser.add_argument(
        "--postgres",
        required=True,
        help="PostgreSQL URL (e.g. postgresql://user:pass@localhost:5432/biosentinel)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify row counts match after migration"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per batch (default: 500)"
    )

    args = parser.parse_args()

    # Normalise postgres URL
    pg_url = args.postgres
    if not pg_url.startswith("postgresql://") and not pg_url.startswith("postgresql+psycopg2://"):
        if pg_url.startswith("postgres://"):
            pg_url = pg_url.replace("postgres://", "postgresql://", 1)
        else:
            pg_url = "postgresql://" + pg_url

    if args.verify:
        run_verify(args.sqlite, pg_url)
    else:
        run_migration(args.sqlite, pg_url, batch_size=args.batch_size)
