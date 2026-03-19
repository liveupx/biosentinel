#!/bin/bash
clear
echo "============================================"
echo " BioSentinel v2.3 — AI Health Monitor"
echo " by Liveupx Pvt. Ltd."
echo "============================================"
echo ""
cd "$(dirname "$0")"
if command -v python3 &>/dev/null; then
    python3 run.py
elif command -v python &>/dev/null; then
    python run.py
else
    echo "ERROR: Python not installed."
    echo "Install from: https://python.org/downloads"
    read -p "Press Enter to exit..."
fi
