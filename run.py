#!/usr/bin/env python3
"""
BioSentinel v2.3 — One-Click Launcher
======================================
Double-click this file, or run:  python run.py

This script:
  1. Checks Python version (3.10+ required)
  2. Auto-installs all required packages (one-time, ~60 seconds)
  3. Trains the ML models (one-time, ~30 seconds)
  4. Seeds 5 demo patients automatically
  5. Opens the API docs in your browser
  6. Shows you how to open the dashboard
"""

import sys, os, subprocess, time, webbrowser, threading, importlib

MIN_PYTHON = (3, 10)
PORT = 8000
URL  = f"http://localhost:{PORT}"

PACKAGES = [
    ("fastapi",     "fastapi==0.110.0"),
    ("uvicorn",     "uvicorn[standard]==0.27.1"),
    ("sqlalchemy",  "sqlalchemy==2.0.28"),
    ("pydantic",    "pydantic==2.6.3"),
    ("jose",        "python-jose[cryptography]==3.3.0"),
    ("passlib",     "passlib[bcrypt]==1.7.4"),
    ("sklearn",     "scikit-learn==1.4.1"),
    ("numpy",       "numpy==1.26.4"),
    ("multipart",   "python-multipart==0.0.9"),
    # OCR — for PDF/image lab report upload feature
    ("pdfplumber",  "pdfplumber==0.11.0"),
    ("pytesseract", "pytesseract==0.3.10"),
    ("PIL",         "Pillow==10.3.0"),
]

# Note: PDF/Image OCR also requires Tesseract system binary.
# Mac:   brew install tesseract
# Linux: sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

def banner():
    print("\n" + "="*56)
    print("  ██████╗ ██╗ ██████╗ ███████╗███████╗███╗   ██╗")
    print("  ██╔══██╗██║██╔═══██╗██╔════╝██╔════╝████╗  ██║")
    print("  ██████╔╝██║██║   ██║███████╗█████╗  ██╔██╗ ██║")
    print("  ██╔══██╗██║██║   ██║╚════██║██╔══╝  ██║╚██╗██║")
    print("  ██████╔╝██║╚██████╔╝███████║███████╗██║ ╚████║")
    print("  ╚═════╝ ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝")
    print("  AI Early Disease Detection v2.0")
    print("  Liveupx Pvt. Ltd. | github.com/liveupx/biosentinel")
    print("="*56 + "\n")

def check_python():
    if sys.version_info < MIN_PYTHON:
        print(f"❌  Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.")
        print(f"   You have {sys.version_info[0]}.{sys.version_info[1]}")
        print("   Download from: https://python.org/downloads")
        input("\nPress Enter to exit..."); sys.exit(1)
    print(f"✓  Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")

def install_packages():
    missing = []
    for import_name, pip_name in PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"✓  {import_name}")
        except ImportError:
            print(f"↓  {pip_name} (will install)")
            missing.append(pip_name)

    if missing:
        print(f"\n📦 Installing {len(missing)} package(s)...\n")
        flags = [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        result = subprocess.run(flags, capture_output=True, text=True)
        if result.returncode != 0:
            flags.append("--break-system-packages")
            result = subprocess.run(flags, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Could not auto-install. Run manually:")
            print(f"   pip install {' '.join(missing)}")
            input("\nPress Enter to exit..."); sys.exit(1)
        print("✅ All packages installed!\n")
    else:
        print("\n✅ All packages ready!\n")

def open_browser():
    import urllib.request
    for _ in range(40):
        time.sleep(1)
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            print(f"\n{'='*56}")
            print(f"  🚀 BioSentinel is running!")
            print(f"")
            print(f"  📊 API docs:  {URL}/docs")
            print(f"  🖥  Dashboard: open biosentinel_dashboard.html in browser")
            print(f"  👤 Patient portal: open biosentinel_patient_portal.html")
            print(f"  👤 Patient view (legacy): open biosentinel_patient_view.html")
            print(f"")
            print(f"  Login: admin / admin123")
            print(f"         dr_sharma / doctor123")
            print(f"")
            print(f"  Press Ctrl+C to stop")
            print(f"{'='*56}\n")
            webbrowser.open(URL + "/docs")
            return
        except Exception:
            continue
    print(f"  Open manually: {URL}/docs")

def main():
    banner()
    check_python()
    print()
    install_packages()

    # Load .env if present
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        print("✓  Loading .env configuration")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    threading.Thread(target=open_browser, daemon=True).start()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    try:
        subprocess.run([sys.executable, script],
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        print("\n\n  👋 BioSentinel stopped. Goodbye!\n")

if __name__ == "__main__":
    main()
