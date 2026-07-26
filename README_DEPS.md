# StreamAudio Dependencies Installation

StreamAudio requires additional dependencies that are NOT in requirements.txt:

```bash
pip install piper-tts
pip install audiotsm
pip install perlin-noise
pip install rich
pip install scipy
```

**OR** just disable TTS if not needed - the app works fine without it.

To add these to your venv:
```bash
.\venv\Scripts\activate
pip install piper-tts audiotsm perlin-noise rich scipy
```

The warning "StreamAudio library not found" means one of these is missing.

## Hosting StreamAudio: the hot daemon

StreamAudio keeps its TTS daemon **hot** between calls -- the Voice model stays
loaded in a long-lived background process so synthesis is fast. That process:

- is **not** a child of whichever program is currently speaking
- often outlives the program that started it, and may have been started by an
  earlier run entirely
- has its own PID, recorded in `tts_daemon.pid`

Any host that manipulates other applications' audio (ducking, mixing, muting)
**must exclude these PIDs**, or it will turn down the very voice it is
producing. Do not assume the daemon shares your PID or your process tree.

```python
from StreamAudio.main import audio_pids

skip = {os.getpid()} | audio_pids()
for session in mixer.sessions():
    if session.pid in skip:
        continue          # this is our own voice
    duck(session)
```

`audio_pids()` returns an empty set when no daemon is running, and re-reads the
PID file on every call so a daemon restart is picked up. Treat the result as
advisory: a daemon killed without running its atexit hook can leave a stale PID.

Aether hit exactly this -- it skipped only `os.getpid()`, ducked the daemon,
and left its own speech quiet in the Windows mixer with no restore.
