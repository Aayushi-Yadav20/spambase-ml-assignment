#!/usr/bin/env bash
# One-shot setup for BITS Virtual Lab (Fedora/RHEL).
# Usage after clone:
#   bash run_bits_lab.sh
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Starting Streamlit at http://localhost:8501 ..."
streamlit run app.py --server.headless true
