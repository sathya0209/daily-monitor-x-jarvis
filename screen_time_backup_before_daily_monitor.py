
import ctypes
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import psutil

# Project location: PersonalAI/src/monitoring/screen_time.py
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "routine.db"

SAMPLE_SECONDS = 5
IDLE_THRESHOLD_SECONDS = 60


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


def get_idle_seconds():
    """Return seconds since the last keyboard or mouse input."""
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)

    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        raise ctypes.WinError()

    # dwTime and GetTickCount are 32-bit millisecond counters.
    elapsed_ms = (
        ctypes.windll.kernel32.GetTickCount() - info.dwTime
    ) & 0xFFFFFFFF

    return elapsed_ms / 1000.0


def get_foreground_app():
    """
    Return the foreground process name only.
    Never read window titles, URLs, or document contents.
    """
    hwnd = ctypes.windll.user32.GetForegroundWindow()

    if not hwnd:
        return "unknown"

    process_id = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(
        hwnd, ctypes.byref(process_id)
    )

    try:
        return psutil.Process(process_id.value).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "unknown"


def init_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screen_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                app_name TEXT NOT NULL,
                state TEXT NOT NULL
                    CHECK (state IN ('active', 'idle')),
                duration_seconds INTEGER NOT NULL
                    CHECK (duration_seconds >= 0)
            )
        """)


def save_segment(start_time, end_time, app_name, state):
    duration = max(
        0,
        int((end_time - start_time).total_seconds())
    )

    if duration == 0:
        return

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
            INSERT INTO screen_activity
            (start_time, end_time, app_name, state, duration_seconds)
            VALUES (?, ?, ?, ?, ?)
        """, (
            start_time.isoformat(timespec="seconds"),
            end_time.isoformat(timespec="seconds"),
            app_name,
            state,
            duration,
        ))


def get_current_activity():
    idle_seconds = get_idle_seconds()

    if idle_seconds >= IDLE_THRESHOLD_SECONDS:
        return ("idle", "IDLE")

    return ("active", get_foreground_app())


def run_monitor(stop_event=None):
    """Run activity tracking until stop_event is set, saving every 30 seconds.

    The optional event lets the desktop assistant start and stop this monitor
    in a background thread without creating a second process.
    """
    if __import__("sys").platform != "win32":
        raise SystemExit("This monitor requires Windows.")

    init_database()

    print("PersonalAI screen-time monitor started.")
    print(f"Database: {DB_PATH}")
    print(f"Sampling interval: {SAMPLE_SECONDS}s")
    print(f"Periodic save interval: 30s")
    print(f"Idle threshold: {IDLE_THRESHOLD_SECONDS}s")
    print("Recording process names and active/idle duration only.")

    previous_key = get_current_activity()
    segment_start = datetime.now()

    try:
        while stop_event is None or not stop_event.is_set():
            if stop_event is None:
                time.sleep(SAMPLE_SECONDS)
            elif stop_event.wait(SAMPLE_SECONDS):
                break

            now = datetime.now()
            current_key = get_current_activity()

            if current_key != previous_key:
                state, app_name = previous_key
                save_segment(segment_start, now, app_name, state)
                print(
                    f"{state.upper():6} | {app_name} | "
                    f"{(now - segment_start).total_seconds():.0f}s"
                )
                previous_key = current_key
                segment_start = now

            elif (now - segment_start).total_seconds() >= 30:
                # Periodic flush prevents a long unchanged app segment from
                # remaining only in memory for the whole session.
                state, app_name = previous_key
                save_segment(segment_start, now, app_name, state)
                print(
                    f"SAVED  | {app_name} [{state}] | "
                    f"{(now - segment_start).total_seconds():.0f}s"
                )
                segment_start = now

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        now = datetime.now()
        state, app_name = previous_key
        save_segment(segment_start, now, app_name, state)
        print("Monitor stopped. Final segment saved.")


def main():
    print("Press Ctrl+C to stop the monitor.")
    run_monitor()


if __name__ == "__main__":
    main()
