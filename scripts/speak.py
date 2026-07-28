# SPDX-License-Identifier: LicenseRef-DNetLab-SourceAvailable-1.0
# SPDX-FileCopyrightText: Copyright (c) 2026 D-Net Lab <https://lab.dnet.live>
#
# Part of StreamAudio, created and maintained by D-Net Lab.
# Free for non-commercial use; commercial use requires a licence.
# Attribution to D-Net Lab is required and may not be removed. See LICENSE.
import sys
import json
import time
import subprocess
import os
from pathlib import Path
from multiprocessing.connection import Client

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from main import ADDRESS, AUTH_KEY, is_server_running, PID_FILE, cleanup_pid_file

def main():
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else "{}"
    cleaned_arg = raw_arg.strip("'\"")
    
    if cleaned_arg == "--clear":
        if is_server_running():
            try:
                with Client(ADDRESS, authkey=AUTH_KEY) as conn:
                    conn.send('--SHUTDOWN--')
                    if conn.poll(5):
                        conn.recv()
            except Exception:
                cleanup_pid_file()
        return

    try:
        payload = json.loads(cleaned_arg)
    except json.JSONDecodeError:
        payload = {"message": cleaned_arg}

    message = payload.get("message") or payload.get("text") or "No message provided."
    message += " [bytes]"

    if not is_server_running():
        # Start the daemon in the background
        subprocess.Popen(
            [sys.executable, os.path.join(str(parent_dir), "main.py"), '--daemon'],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Poll for it to be ready
        ready = False
        for _ in range(30):
            time.sleep(0.5)
            if is_server_running():
                ready = True
                break
                
        if not ready:
            print("Error: Could not start or connect to TTS daemon.", file=sys.stderr)
            return

    try:
        with Client(ADDRESS, authkey=AUTH_KEY) as conn:
            conn.send(message)
            if conn.poll(60): # Wait up to 60s for synthesis
                res = conn.recv()
                print(f"###{res}###")
    except Exception as e:
        print(f"Error communicating with TTS service: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()