#!/usr/bin/env bash
#
# One-command setup + run.
# Creates an isolated virtual environment (.venv), installs the pinned
# dependencies from requirements.txt, and runs the full pipeline.
#
# Prerequisite: place the Kaggle dataset at data/marketing_campaign.csv first
# (it is not auto-downloaded — Kaggle requires an account). See README.md.
#
# Usage:  bash run.sh
set -e

# Fail fast (before installing anything) if the dataset is missing.
if [ ! -f "data/marketing_campaign.csv" ]; then
    echo "ERROR: data/marketing_campaign.csv not found."
    echo "Download 'Customer Personality Analysis' from Kaggle:"
    echo "  https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis"
    echo "and place marketing_campaign.csv in the data/ folder, then re-run."
    exit 1
fi

# Create the virtual environment only if it does not already exist.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv
fi

# Activate it and install the exact pinned dependencies.
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the full pipeline.
python main.py
