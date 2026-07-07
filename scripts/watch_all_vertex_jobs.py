#!/usr/bin/env python3
"""
scripts/watch_all_vertex_jobs.py
Background daemon that polls and logs the status of the 6 running Llama SFT jobs.
"""

import subprocess
import time
import sys
from pathlib import Path

REGION = "us-central1"
LOG_FILE = Path("/home/forge/pbt_vertex_status.log")

JOBS = [
    "pbt-llama3_1-8b-treatment-seed17",
    "pbt-gemma4-26b-treatment-seed42" if False else "pbt-llama3_1-8b-treatment-seed42",
    "pbt-llama3_1-8b-treatment-seed42" if False else "pbt-llama3_1-8b-treatment-seed73",  # map to llama
    "pbt-llama3_1-8b-control-seed17",
    "pbt-llama3_1-8b-control-seed42",
    "pbt-llama3_1-8b-control-seed73"
]

def log_message(msg: str):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

log_message("Watchdog daemon initialized cleanly.")

# Ensure we have a clean status file starting up
if LOG_FILE.exists():
    LOG_FILE.unlink()

while True:
    try:
        # Get list of jobs from gcloud
        res = subprocess.run(
            ["gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", "--limit=15", "--format=value(displayName, state)"],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = res.stdout.strip().split("\n")
        states = {}
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                states[parts[0]] = parts[1]
                
        # Format a summary block
        active_found = False
        summary = []
        for j in JOBS:
            # Find the most recent matching job by name prefix
            matching_state = "UNKNOWN"
            for disp_name, state in states.items():
                if disp_name.startswith(j):
                    matching_state = state
                    active_found = True
                    break
            summary.append(f"  {j:<45s} : {matching_state}")
            
        log_message(f"--- STATUS CHECK ---")
        for s in summary:
            log_message(s)
            
        # Check if all jobs reached terminal state
        terminal_states = ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]
        all_done = True
        for j in JOBS:
            matching_state = "UNKNOWN"
            for disp_name, state in states.items():
                if disp_name.startswith(j):
                    matching_state = state
                    break
            if matching_state not in terminal_states:
                all_done = False
                
        if all_done and active_found:
            log_message("All jobs have reached a terminal state. Exiting watchdog.")
            break
            
    except Exception as e:
        log_message(f"Polling error: {repr(e)}")
        
    time.sleep(60)
