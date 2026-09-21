#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "[*] Installing Python requirements..."
pip install -r requirements.txt

echo "[*] Pre-caching neural network weights for fast inference..."
python preload_models.py

echo "[+] Build completed successfully!"
