#!/usr/bin/env python3
"""
Smart Dataset Downloader for Supertonic v2 TTS

Downloads a fraction of EuroSpeech Ukrainian dataset.
Set HF_TOKEN environment variable before running!
"""

import os
from huggingface_hub import HfApi, snapshot_download

# Configuration
REPO_ID = "disco-eth/EuroSpeech"
SUBSET_FOLDER = "ukraine"
FRACTION = 0.25  # Download only 25% of files (~300 hours from 1200)

# Enable fast transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Get token from environment (NEVER hardcode!)
token = os.environ.get("HF_TOKEN")
if not token:
    print("⚠️  Warning: HF_TOKEN not set. Some datasets may require authentication.")
    print("   Set it with: export HF_TOKEN=your_token_here")

print(f"🔍 Getting file list from {REPO_ID}...")
api = HfApi(token=token)

# 1. Get all repository files
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

# 2. Filter only Ukrainian files
uk_files = [f for f in all_files if f.startswith(SUBSET_FOLDER)]
uk_files.sort()  # Sort to take in order (data_000, data_001...)

total_files = len(uk_files)
files_to_download = int(total_files * FRACTION)
target_files = uk_files[:files_to_download]

print(f"📊 Total Ukrainian files: {total_files}")
print(f"✂️  Will download first: {files_to_download} ({int(FRACTION*100)}% of dataset)")

# 3. Download only selected files
print(f"⬇️  Starting download...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir="data/raw/eurospeech",
    allow_patterns=target_files,
    token=token
)

print("✅ Download complete!")
