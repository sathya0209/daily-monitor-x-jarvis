
import os
import sys
import sqlite3
import subprocess
import asyncio
import base64
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"

DB_PATH = DATA_DIR / "routine.db"
MONITOR_SCRIPT = BASE_DIR / "src" / "monitoring" / "screen_time.py"
PID_PATH = DATA_DIR / "screen_monitor.pid"
LOG_PATH = LOG_DIR / "screen_monitor.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Personal AI Routine Assistant",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 2. DATABASE
# ============================================================

def db_connection():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    with db_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                category TEXT NOT NULL,
                minutes INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                entry_date TEXT NOT NULL
            )
        """)

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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_categories (
                app_name TEXT PRIMARY KEY,
                category TEXT NOT NULL
            )
        """)


def get_tasks():
    with db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT id, task, category, minutes, completed, entry_date
            FROM tasks
            ORDER BY id DESC
            """,
            conn,
        )


def add_task(task, category, minutes):
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO tasks
                (task, category, minutes, completed, entry_date)
            VALUES (?, ?, ?, 0, ?)
            """,
            (
                task,
                category,
                minutes,
                date.today().isoformat(),
            ),
        )


def set_completed(task_id, completed):
    with db_connection() as conn:
        conn.execute(
            "UPDATE tasks SET completed = ? WHERE id = ?",
            (int(completed), int(task_id)),
        )


def get_screen_data(days=7):
    cutoff = (
        date.today() - timedelta(days=days - 1)
    ).isoformat()

    with db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT id, start_time, end_time,
                   app_name, state, duration_seconds
            FROM screen_activity
            WHERE date(start_time) >= date(?)
            ORDER BY start_time DESC
            """,
            conn,
            params=(cutoff,),
        )


def get_categories():
    with db_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT app_name, category
            FROM app_categories
            ORDER BY app_name
            """,
            conn,
        )


def save_category(app_name, category):
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_categories (app_name, category)
            VALUES (?, ?)
            ON CONFLICT(app_name)
            DO UPDATE SET category = excluded.category
            """,
            (app_name, category),
        )


# ============================================================
# 3. MONITOR PROCESS MANAGEMENT
# ============================================================

def get_monitor_process():
    """
    Find the monitor process using its PID file.
    Verify the process command line to avoid accepting a reused PID.
    """
    if not PID_PATH.exists():
        return None

    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        process = psutil.Process(pid)

        if not process.is_running():
            PID_PATH.unlink(missing_ok=True)
            return None

        command_line = " ".join(process.cmdline()).lower()
        expected_script = str(MONITOR_SCRIPT).lower()

        if expected_script in command_line:
            return process

    except (ValueError, OSError, psutil.Error):
        pass

    PID_PATH.unlink(missing_ok=True)
    return None


def start_monitor():
    if not MONITOR_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Monitor script not found: {MONITOR_SCRIPT}"
        )

    if get_monitor_process():
        return

    # Clear an old PID file before starting a new process.
    PID_PATH.unlink(missing_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, str(MONITOR_SCRIPT)],
            cwd=str(BASE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            ),
            close_fds=True,
        )

    PID_PATH.write_text(str(process.pid), encoding="utf-8")


def stop_monitor():
    """
    Stop the monitor process.

    Note: terminate() is abrupt. The monitor script should be updated
    separately to support graceful shutdown and save its final segment.
    """
    process = get_monitor_process()

    if not process:
        PID_PATH.unlink(missing_ok=True)
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=3)
        except psutil.Error:
            pass

    PID_PATH.unlink(missing_ok=True)


# ============================================================
# 4. GROQ AI ANALYSIS
# ============================================================

def generate_ai_review(question, tasks, screen_data):

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    if not api_key:
        return (
            "Groq API key not found. Check the GROQ_API_KEY "
            "setting in your .env file."
        )

    if not model:
        return (
            "Groq model not configured. Add GROQ_MODEL to .env "
            "using the exact model ID that worked previously."
        )

    cutoff = (
        date.today() - timedelta(days=6)
    ).isoformat()

    recent_tasks = tasks[
        tasks["entry_date"] >= cutoff
    ].copy()

    task_columns = [
        "task", "category", "minutes", "completed", "entry_date"
    ]

    task_records = (
        recent_tasks[task_columns].to_dict(orient="records")
        if not recent_tasks.empty
        else []
    )

    # Summarize screen usage before sending it to Groq.
    # This avoids sending an unnecessarily large raw history.
    screen_summary = []

    if not screen_data.empty:
        screen_data = screen_data.copy()

        screen_data["start_time"] = pd.to_datetime(
            screen_data["start_time"], errors="coerce"
        )

        screen_data["day"] = (
            screen_data["start_time"].dt.strftime("%Y-%m-%d")
        )

        summary = (
            screen_data.groupby(
                ["day", "app_name", "state"], as_index=False
            )["duration_seconds"].sum()
        )

        categories = get_categories()

        summary = summary.merge(
            categories,
            on="app_name",
            how="left",
        )

        summary["category"] = (
            summary["category"].fillna("Uncategorized")
        )

        summary["minutes"] = (
            summary["duration_seconds"] / 60
        ).round(1)

        screen_summary = summary[
            ["day", "app_name", "category", "state", "minutes"]
        ].to_dict(orient="records")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a personalized productivity coach. "
                    "Analyze only the supplied task and application "
                    "usage records. Distinguish measured facts from "
                    "inferences. Application usage does not prove "
                    "productivity. A browser, terminal, or editor may "
                    "be used for work or non-work. Do not invent data. "
                    "Compare logged task completion with screen usage "
                    "where meaningful. Identify missing data, patterns, "
                    "and up to three realistic improvements. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Today's date: {date.today().isoformat()}\n\n"
                    f"Task records from the last seven days:\n"
                    f"{task_records}\n\n"
                    f"Summarized screen activity from the last "
                    f"seven days:\n{screen_summary}\n\n"
                    f"My question:\n{question}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return (
        response.choices[0].message.content
        or "The AI returned an empty response."
    )


# ============================================================
# 4B. JARVIS VOICE HELPERS
# ============================================================

def jarvis_transcribe_audio(audio_bytes, language_mode):
    """Transcribe a browser-recorded WAV file with Groq Whisper."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    # A temporary WAV file makes the audio format explicit for Groq.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(audio_bytes)
        temp_path = temp.name

    try:
        with open(temp_path, "rb") as audio_file:
            args = {
                "file": audio_file,
                "model": "whisper-large-v3-turbo",
                "response_format": "text",
                "temperature": 0.0,
            }
            if language_mode == "English":
                args["language"] = "en"
                args["prompt"] = (
                    "Indian English speech. Preserve names and technical terms."
                )
            else:
                # Do not force Tamil or English for mixed Tanglish speech.
                args["prompt"] = (
                    "Spoken Chennai Tamil mixed with English, known as Tanglish. "
                    "The speaker may use Tamil words, English words, and technical "
                    "terms. Transcribe only what was spoken; do not add words."
                )
            result = Groq(api_key=api_key).audio.transcriptions.create(**args)

        if isinstance(result, str):
            return result.strip()
        return getattr(result, "text", "").strip()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def jarvis_make_speech_audio(text, language_mode):
    """Generate neural speech audio. Returns MP3 bytes."""
    import edge_tts

    speech_text = text
    if language_mode == "Tanglish":
        # Convert the displayed romanized Tanglish to spoken Tamil script.
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        if api_key:
            try:
                response = Groq(api_key=api_key).chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Convert this Chennai-style Tanglish answer into "
                                "natural, modern spoken Tamil script for a Tamil "
                                "text-to-speech voice. Preserve meaning. Do not add "
                                "facts or sentences. Keep common English technical "
                                "terms and names in English when natural. Return "
                                "only the speech text."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                    max_completion_tokens=800,
                )
                converted = response.choices[0].message.content
                if converted and converted.strip():
                    speech_text = converted.strip()
            except Exception:
                # If conversion fails, speak the original answer.
                speech_text = text

    voice = (
        "ta-IN-PallaviNeural"
        if language_mode == "Tanglish"
        else "en-IN-NeerjaNeural"
    )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp:
        audio_path = temp.name

    try:
        async def create_audio():
            communicator = edge_tts.Communicate(speech_text, voice)
            await communicator.save(audio_path)

        asyncio.run(create_audio())
        with open(audio_path, "rb") as audio_file:
            return audio_file.read()
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def jarvis_autoplay_audio(audio_bytes):
    """Render speech with Streamlit's native audio player to avoid iframe clipping."""
    if not audio_bytes:
        return
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)


def jarvis_local_intent(question, language_mode, tasks, screen_data):
    """Handle dashboard-aware requests locally before using general chat AI."""
    import re

    q = question.strip()
    q_lower = q.lower()
    today_iso = date.today().isoformat()
    today_tasks = tasks[tasks["entry_date"] == today_iso].copy() if not tasks.empty else tasks.copy()

    # Create a real task in the existing SQLite database.
    create_words = ("add task", "create task", "add a task", "create a task",
                    "add activity", "create activity", "remind me to")
    if any(word in q_lower for word in create_words):
        task_text = q
        task_text = re.sub(r"(?i)^.*?(?:add task|create task|add a task|create a task|add activity|create activity|remind me to)\s*[:,-]?\s*", "", task_text).strip()
        if not task_text or task_text.lower() in {"today", "for today", "a task"}:
            return ("Please say the task name, for example: Add task: finish Python assignment, 45 minutes, Study." if language_mode == "English" else "Enna task add pannanum nu sollunga. Example: Add task: finish Python assignment, 45 minutes, Study.")
        minutes_match = re.search(r"(?i)(\d+)\s*(?:minutes?|mins?|min)\b", task_text)
        minutes = int(minutes_match.group(1)) if minutes_match else 30
        minutes = max(0, min(minutes, 1440))
        category = "Personal"
        for candidate in ["Study", "Work", "Habit", "Exercise", "Personal"]:
            if re.search(rf"(?i)\b{candidate}\b", task_text):
                category = candidate
                break
        cleaned = re.sub(r"(?i)[,;]?\s*\d+\s*(?:minutes?|mins?|min)\b", "", task_text)
        cleaned = re.sub(r"(?i)[,;]?\s*(?:Study|Work|Habit|Exercise|Personal)\s*$", "", cleaned).strip(" ,;:-")
        if not cleaned:
            cleaned = task_text.strip(" ,;:-")
        add_task(cleaned, category, minutes)
        return (f"Done! I added '{cleaned}' to today's tasks under {category}, planned for {minutes} minutes." if language_mode == "English" else f"Seri! '{cleaned}' task innikku add panniten. Category {category}, plan time {minutes} minutes.")

    # Today's tasks and completion status.
    if any(word in q_lower for word in ["my tasks", "show tasks", "pending task", "pending tasks", "today's task", "todays task", "tasks today", "task list"]):
        if today_tasks.empty:
            return ("You don't have any tasks recorded for today yet. Say 'Add task: ...' and I'll create one." if language_mode == "English" else "Innikku tasks edhuvum record pannala. 'Add task: ...' nu sollunga, naan add pannuren.")
        lines = []
        for _, row in today_tasks.head(12).iterrows():
            status = "Done" if int(row["completed"]) else "Pending"
            lines.append(f"• {row['task']} — {status} — {row['minutes']} min ({row['category']})")
        completed = int(today_tasks["completed"].sum())
        heading = f"Today's tasks: {completed}/{len(today_tasks)} completed.\n" if language_mode == "English" else f"Innikku tasks: {completed}/{len(today_tasks)} complete.\n"
        return heading + "\n".join(lines)

    # Today's recorded activity and application usage.
    if any(word in q_lower for word in ["today's activity", "todays activity", "today activity", "screen time today", "today's screen time", "todays screen time", "what did i do today", "my activity today"]):
        screen_today = screen_data.copy()
        if not screen_today.empty:
            screen_today["start_time"] = pd.to_datetime(screen_today["start_time"], errors="coerce")
            screen_today = screen_today[screen_today["start_time"].dt.date == date.today()]
        active = int(screen_today.loc[screen_today["state"] == "active", "duration_seconds"].sum()) if not screen_today.empty else 0
        idle = int(screen_today.loc[screen_today["state"] == "idle", "duration_seconds"].sum()) if not screen_today.empty else 0
        if screen_today.empty:
            app_lines = "No screen activity has been recorded today yet. Make sure tracking is running." if language_mode == "English" else "Innikku screen activity innum record aagala. Tracking running-la irukka check pannunga."
        else:
            usage = screen_today[screen_today["state"] == "active"].groupby("app_name")["duration_seconds"].sum().sort_values(ascending=False).head(5)
            app_lines = "Top active apps: " + ", ".join(f"{name} ({seconds/60:.0f} min)" for name, seconds in usage.items()) if not usage.empty else "No active application usage recorded."
        task_line = f"Tasks: {int(today_tasks['completed'].sum())}/{len(today_tasks)} completed." if not today_tasks.empty else "Tasks: none recorded today."
        if language_mode == "English":
            return f"Today's recorded activity\nActive screen time: {active/60:.1f} minutes\nIdle time: {idle/60:.1f} minutes\n{task_line}\n{app_lines}\n\nNote: app time shows foreground usage, not proof of productivity."
        return f"Innikku recorded activity\nActive screen time: {active/60:.1f} minutes\nIdle time: {idle/60:.1f} minutes\n{task_line}\n{app_lines}\nApp usage mattum productivity-ku proof illa."

    # Use the existing productivity coach with real task and screen records.
    if any(word in q_lower for word in ["productivity", "analyze my day", "analyse my day", "productivity review", "improve my focus", "how productive"]):
        return generate_ai_review(q, tasks, screen_data)

    return None


def jarvis_answer(question, language_mode, tasks=None, screen_data=None):
    """Answer using local dashboard data for supported intents, otherwise Groq chat."""
    tasks = tasks if tasks is not None else get_tasks()
    screen_data = screen_data if screen_data is not None else get_screen_data(days=7)
    local_answer = jarvis_local_intent(question, language_mode, tasks, screen_data)
    if local_answer is not None:
        return local_answer

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    language_instruction = (
        "Reply in natural, modern Chennai-style Tanglish using English letters. "
        "Mix Tamil and English as people do in everyday conversation. Avoid formal "
        "Tamil. Be accurate and concise."
        if language_mode == "Tanglish"
        else "Reply in clear, conversational Indian English. Be accurate and concise."
    )

    response = Groq(api_key=api_key).chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are JARVIS, a friendly personal assistant. Make answers "
                    "easy to understand when spoken aloud. Do not invent facts or "
                    "claim to have performed actions you did not perform. "
                    + language_instruction
                ),
            },
            {"role": "user", "content": question.strip()},
        ],
        temperature=0.2,
        max_completion_tokens=800,
    )
    return (
        response.choices[0].message.content or "I couldn't generate a response."
    ).strip()


# ============================================================
# 5. INITIALIZATION AND AUTOMATIC REFRESH
# ============================================================

init_db()

# Refresh the dashboard every 60 seconds so generated speech is not interrupted.
st_autorefresh(
    interval=60_000,
    key="personal_ai_auto_refresh",
)


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.title("Personal AI Routine Assistant")

st.caption(
    "Local SQLite storage | Windows screen-time monitoring | Groq AI"
)


# ============================================================
# 7. LIVE TRACKING CONTROLS
# ============================================================

st.header("1. Windows screen-time tracking")

monitor = get_monitor_process()
is_running = monitor is not None

status_col, start_col, stop_col = st.columns([2, 1, 1])

with status_col:

    if is_running:
        st.success("Tracking is running")
    else:
        st.warning("Tracking is stopped")

    st.caption(
        f"Dashboard refreshed at {datetime.now().strftime('%H:%M:%S')}"
    )

with start_col:

    if st.button(
        "▶ Start tracking",
        disabled=is_running,
        use_container_width=True,
    ):
        try:
            start_monitor()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not start monitor: {exc}")

with stop_col:

    if st.button(
        "■ Stop tracking",
        disabled=not is_running,
        use_container_width=True,
    ):
        try:
            stop_monitor()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not stop monitor: {exc}")

with st.expander("Monitor diagnostics"):

    st.write(f"Monitor script: `{MONITOR_SCRIPT}`")
    st.write(f"SQLite database: `{DB_PATH}`")
    st.write(f"Monitor log: `{LOG_PATH}`")

    if st.button("Read monitor log"):

        if LOG_PATH.exists():
            log_text = LOG_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            )

            st.code(log_text[-6000:] or "Log file is empty.")
        else:
            st.info("The monitor log has not been created yet.")


# ============================================================
# 8. TASK MANAGEMENT
# ============================================================

st.divider()
st.header("2. Daily tasks and habits")

with st.form("add_task_form", clear_on_submit=True):

    task = st.text_input("Task or activity")

    task_category = st.selectbox(
        "Task category",
        ["Study", "Work", "Habit", "Exercise", "Personal"],
    )

    minutes = st.number_input(
        "Duration in minutes",
        min_value=0,
        max_value=1440,
        value=30,
    )

    submitted = st.form_submit_button("Add activity")

    if submitted:

        if task.strip():
            add_task(
                task.strip(),
                task_category,
                int(minutes),
            )

            st.success("Activity saved.")
            st.rerun()

        else:
            st.warning("Please enter a task name.")


tasks = get_tasks()

if tasks.empty:

    st.info("No tasks recorded yet. Add your first activity above.")

else:

    today_tasks = tasks[
        tasks["entry_date"] == date.today().isoformat()
    ]

    total_tasks = len(today_tasks)
    completed_tasks = int(today_tasks["completed"].sum())

    completion_rate = (
        completed_tasks / total_tasks * 100
        if total_tasks
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Today's activities", total_tasks)
    col2.metric("Completed", completed_tasks)
    col3.metric("Completion rate", f"{completion_rate:.0f}%")

    st.subheader("Activity history")

    for _, row in tasks.iterrows():

        label = (
            f"{row['task']} | {row['category']} | "
            f"{row['minutes']} min | {row['entry_date']}"
        )

        checked = st.checkbox(
            label,
            value=bool(row["completed"]),
            key=f"task_{row['id']}",
        )

        if checked != bool(row["completed"]):
            set_completed(row["id"], checked)
            st.rerun()

    # Refresh task data after any checkbox changes.
    tasks = get_tasks()

    st.download_button(
        "Download task history CSV",
        data=tasks.to_csv(index=False).encode("utf-8"),
        file_name="routine_history.csv",
        mime="text/csv",
    )


# ============================================================
# 9. DAILY AND WEEKLY SCREEN-TIME REPORTS
# ============================================================

st.divider()
st.header("3. Daily and weekly screen-time analytics")

screen_data = get_screen_data(days=7)

if screen_data.empty:

    st.info(
        "No screen-time records were found for the last seven days. "
        "Start tracking and use your computer to collect data."
    )

else:

    screen_data = screen_data.copy()

    screen_data["start_time"] = pd.to_datetime(
        screen_data["start_time"],
        errors="coerce",
    )

    screen_data["minutes"] = (
        screen_data["duration_seconds"] / 60
    )

    today_data = screen_data[
        screen_data["start_time"].dt.date == date.today()
    ].copy()

    active_today = today_data.loc[
        today_data["state"] == "active",
        "duration_seconds",
    ].sum()

    idle_today = today_data.loc[
        today_data["state"] == "idle",
        "duration_seconds",
    ].sum()

    active_week = screen_data.loc[
        screen_data["state"] == "active",
        "duration_seconds",
    ].sum()

    idle_week = screen_data.loc[
        screen_data["state"] == "idle",
        "duration_seconds",
    ].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Today active", f"{active_today / 60:.1f} min")
    col2.metric("Today idle", f"{idle_today / 60:.1f} min")
    col3.metric("7-day active", f"{active_week / 3600:.2f} h")
    col4.metric("7-day idle", f"{idle_week / 3600:.2f} h")

    active_data = screen_data[
        screen_data["state"] == "active"
    ].copy()

    if not active_data.empty:

        st.subheader("Active time by application")

        app_usage = (
            active_data.groupby("app_name")["duration_seconds"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(app_usage / 60)

        report = app_usage.reset_index()

        report["minutes"] = (
            report["duration_seconds"] / 60
        ).round(1)

        categories = get_categories()

        report = report.merge(
            categories,
            on="app_name",
            how="left",
        )

        report["category"] = (
            report["category"].fillna("Uncategorized")
        )

        st.dataframe(
            report[["app_name", "category", "minutes"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Daily active and idle trends")

    daily_data = screen_data.copy()

    daily_data["day"] = (
        daily_data["start_time"].dt.strftime("%Y-%m-%d")
    )

    daily_summary = daily_data.pivot_table(
        index="day",
        columns="state",
        values="duration_seconds",
        aggfunc="sum",
        fill_value=0,
    )

    daily_summary = daily_summary / 60

    st.line_chart(daily_summary)


# ============================================================
# 10. APPLICATION CATEGORY CONFIGURATION
# ============================================================

st.divider()
st.header("4. Configure application categories")

st.caption(
    "Choose a category for each application. Categories are saved "
    "locally and can be changed whenever you want."
)

all_screen_data = get_screen_data(days=30)

if all_screen_data.empty:

    st.info(
        "Application categories will be available after "
        "your monitor records some usage."
    )

else:

    app_names = sorted(
        all_screen_data["app_name"].dropna().unique().tolist()
    )

    categories = get_categories()

    category_map = dict(
        zip(categories["app_name"], categories["category"])
    )

    choices = [
        "Productive",
        "Entertainment",
        "Communication",
        "Utility",
        "Other",
    ]

    selected_app = st.selectbox(
        "Select an application",
        app_names,
    )

    current_category = category_map.get(
        selected_app,
        "Other",
    )

    selected_category = st.selectbox(
        "Select its category",
        choices,
        index=(
            choices.index(current_category)
            if current_category in choices
            else choices.index("Other")
        ),
    )

    if st.button("Save application category"):

        save_category(
            selected_app,
            selected_category,
        )

        st.success(
            f"{selected_app} categorized as {selected_category}."
        )

        st.rerun()


# ============================================================
# 11. AI PRODUCTIVITY ANALYSIS
# ============================================================

st.divider()
st.header("5. AI productivity coach")

st.caption(
    "Analyze recent task completion alongside application usage. "
    "The assistant distinguishes recorded facts from estimates."
)

question = st.text_area(
    "What would you like to understand?",
    placeholder=(
        "Example: Compare my study tasks with my application usage "
        "this week and suggest three improvements."
    ),
)

if "ai_review_result" not in st.session_state:
    st.session_state["ai_review_result"] = None

if st.button(
    "Analyze tasks and screen time",
    type="primary",
):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing your tasks and screen time..."):
            st.session_state["ai_review_result"] = generate_ai_review(
                question,
                tasks,
                screen_data,
            )

if st.session_state["ai_review_result"] is not None:
    st.markdown("### AI review")
    st.markdown(st.session_state["ai_review_result"])
# ============================================================
# 12. FLOATING JARVIS CHAT AND VOICE ASSISTANT
# ============================================================

# Floating JARVIS mascot. The closed state is a small robot character button;
# opening it reveals the compact chat panel.
st.markdown(
    """
    <style>
    .st-key-jarvis_float {
        position: fixed;
        right: 22px;
        bottom: 20px;
        z-index: 99999;
        width: 380px;
        max-width: calc(100vw - 28px);
        background: rgba(15, 20, 32, 0.97);
        border: 1px solid rgba(63, 150, 255, 0.65);
        border-radius: 20px;
        padding: 14px;
        box-shadow: 0 0 28px rgba(0, 140, 255, 0.18), 0 12px 40px rgba(0,0,0,0.48);
        backdrop-filter: blur(14px);
    }
    .st-key-jarvis_float [data-testid="stVerticalBlock"] { gap: 0.55rem; }
    .st-key-jarvis_float [data-testid="stChatMessage"] { padding: 0.45rem; }
    .st-key-jarvis_float [data-testid="stAudioInput"] { margin-bottom: 0.2rem; }

    /* When chat is closed, show only the floating robot mascot launcher. */
    .st-key-jarvis_float:has(.st-key-jarvis_open_button) {
        width: auto;
        max-width: none;
        padding: 0;
        background: transparent;
        border: 0;
        box-shadow: none;
        backdrop-filter: none;
    }
    .st-key-jarvis_float .st-key-jarvis_open_button button {
        width: 78px;
        height: 78px;
        min-height: 78px;
        padding: 5px;
        border-radius: 50%;
        border: 2px solid #42baff;
        background: radial-gradient(circle at 35% 25%, #344e78, #10182a 72%);
        color: white;
        font-size: 35px;
        box-shadow: 0 0 0 5px rgba(45, 160, 255, 0.12), 0 0 28px rgba(0, 140, 255, 0.55);
        animation: jarvis-float-glow 2.8s ease-in-out infinite;
    }
    .st-key-jarvis_float .st-key-jarvis_open_button button p {
        font-size: 0 !important;
        line-height: 1 !important;
    }
    .st-key-jarvis_float .st-key-jarvis_open_button button p:before {
        content: "🤖";
        font-size: 35px;
    }
    .st-key-jarvis_float .st-key-jarvis_open_button button:hover {
        border-color: #a2e4ff;
        transform: translateY(-3px) scale(1.04);
    }
    @keyframes jarvis-float-glow {
        0%, 100% { box-shadow: 0 0 0 4px rgba(45,160,255,.10), 0 0 22px rgba(0,140,255,.38); }
        50% { box-shadow: 0 0 0 8px rgba(45,160,255,.08), 0 0 34px rgba(0,180,255,.65); }
    }
    @media (max-width: 480px) {
        .st-key-jarvis_float { right: 10px; bottom: 10px; width: calc(100vw - 20px); }
        .st-key-jarvis_float:has(.st-key-jarvis_open_button) { width: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "jarvis_messages" not in st.session_state:
    st.session_state["jarvis_messages"] = []
if "jarvis_audio_pending" not in st.session_state:
    st.session_state["jarvis_audio_pending"] = None
if "jarvis_transcript" not in st.session_state:
    st.session_state["jarvis_transcript"] = ""
if "jarvis_open" not in st.session_state:
    st.session_state["jarvis_open"] = False

with st.container(key="jarvis_float"):
    if not st.session_state["jarvis_open"]:
        if st.button("🤖", key="jarvis_open_button", type="primary", help="Chat with JARVIS"):
            st.session_state["jarvis_open"] = True
            st.rerun()
    else:
        top_left, top_right = st.columns([4, 1])
        with top_left:
            st.markdown("### 🤖 JARVIS")
            st.caption("Your personal AI assistant · Ready to help")
        with top_right:
            if st.button("✕", key="jarvis_close_button", help="Close JARVIS"):
                st.session_state["jarvis_open"] = False
                st.rerun()

        jarvis_language = st.radio(
            "Response language",
            options=["English", "Tanglish"],
            horizontal=True,
            key="jarvis_language",
            label_visibility="collapsed",
        )

        quick1, quick2, quick3 = st.columns(3)
        with quick1:
            if st.button("Today's activity", key="jarvis_quick_activity", use_container_width=True):
                st.session_state["jarvis_quick_prompt"] = "Show my today's activity and screen time"
        with quick2:
            if st.button("My tasks", key="jarvis_quick_tasks", use_container_width=True):
                st.session_state["jarvis_quick_prompt"] = "Show my tasks today"
        with quick3:
            if st.button("Productivity", key="jarvis_quick_productivity", use_container_width=True):
                st.session_state["jarvis_quick_prompt"] = "Analyze my productivity today and suggest improvements"

        if st.session_state.get("jarvis_quick_prompt"):
            quick_prompt = st.session_state.pop("jarvis_quick_prompt")
            st.session_state["jarvis_messages"].append({"role": "user", "content": quick_prompt})
            try:
                with st.spinner("JARVIS is checking your dashboard..."):
                    quick_answer = jarvis_answer(quick_prompt, jarvis_language, get_tasks(), get_screen_data(days=7))
                st.session_state["jarvis_messages"].append({"role": "assistant", "content": quick_answer})
                try:
                    st.session_state["jarvis_audio_pending"] = jarvis_make_speech_audio(quick_answer, jarvis_language)
                except Exception:
                    st.session_state["jarvis_audio_pending"] = None
            except Exception as exc:
                st.session_state["jarvis_messages"].append({"role": "assistant", "content": f"I couldn't complete that request: {exc}"})
            st.rerun()

        st.markdown("<div style='max-height:250px; overflow-y:auto;'>", unsafe_allow_html=True)
        if not st.session_state["jarvis_messages"]:
            st.info("👋 Hi! I'm JARVIS. I can help with your tasks, today's activity, screen time, and questions.")
        for message in st.session_state["jarvis_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        recorded_audio = st.audio_input("Record a voice message", key="jarvis_audio_input")
        if recorded_audio is not None:
            import hashlib
            audio_signature = hashlib.sha256(recorded_audio.getvalue()).hexdigest()
            if st.session_state.get("jarvis_last_audio_signature") != audio_signature:
                st.session_state["jarvis_last_audio_signature"] = audio_signature
                with st.spinner("Recognizing your speech..."):
                    try:
                        st.session_state["jarvis_transcript"] = jarvis_transcribe_audio(
                            recorded_audio.getvalue(), jarvis_language
                        )
                    except Exception as exc:
                        st.error(f"Speech recognition failed: {exc}")

        if st.session_state["jarvis_transcript"]:
            st.session_state["jarvis_transcript"] = st.text_input(
                "Recognized words (edit if needed)",
                value=st.session_state["jarvis_transcript"],
                key="jarvis_transcript_editor",
            )

        with st.form("jarvis_chat_form", clear_on_submit=True):
            typed_message = st.text_input(
                "Message", placeholder="Ask JARVIS anything...",
                key="jarvis_typed_message", label_visibility="collapsed"
            )
            send_jarvis = st.form_submit_button("Send", type="primary", use_container_width=True)

        clear_col, _ = st.columns([1, 1])
        with clear_col:
            if st.button("Clear chat", key="jarvis_clear_button"):
                st.session_state["jarvis_messages"] = []
                st.session_state["jarvis_transcript"] = ""
                st.session_state["jarvis_audio_pending"] = None
                st.rerun()

        if send_jarvis:
            question_to_send = typed_message.strip() or st.session_state["jarvis_transcript"].strip()
            if not question_to_send:
                st.warning("Type a message or record your voice first.")
            else:
                st.session_state["jarvis_messages"].append({"role": "user", "content": question_to_send})
                try:
                    with st.spinner("JARVIS is thinking..."):
                        answer = jarvis_answer(question_to_send, jarvis_language, get_tasks(), get_screen_data(days=7))
                    st.session_state["jarvis_messages"].append({"role": "assistant", "content": answer})
                    st.session_state["jarvis_transcript"] = ""
                    try:
                        st.session_state["jarvis_audio_pending"] = jarvis_make_speech_audio(answer, jarvis_language)
                    except Exception as speech_exc:
                        st.session_state["jarvis_audio_pending"] = None
                        st.warning(f"JARVIS answered, but speech generation failed: {speech_exc}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"JARVIS could not answer: {exc}")

        if st.session_state.get("jarvis_audio_pending"):
            pending_audio = st.session_state["jarvis_audio_pending"]
            st.session_state["jarvis_audio_pending"] = None
            jarvis_autoplay_audio(pending_audio)

# ============================================================
# 12B. OPTIONAL HANDS-FREE "HEY JARVIS" VOICE COMMAND
# ============================================================
# Enable microphone listening explicitly. In Chrome/Edge, saying
# "Hey JARVIS, show my tasks" opens the popup and submits the command.
components.html(
    r"""
    <div id="jarvis-wake" style="font-family:Arial,sans-serif;">
      <button id="wake-toggle" type="button" style="
        border:1px solid #328cff;border-radius:14px;padding:10px 13px;
        background:#111b2c;color:#dcecff;cursor:pointer;font-size:13px;
        box-shadow:0 0 16px rgba(40,140,255,.22);">
        🎙️ Enable Hey JARVIS
      </button>
      <div id="wake-status" style="font-size:11px;color:#9fb4d3;margin-top:5px;">
        Voice assistant is off. Enable it, then say “Hey JARVIS, …”
      </div>
    </div>
    <script>
    (() => {
      const button = document.getElementById('wake-toggle');
      const status = document.getElementById('wake-status');
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      let recognition = null;
      let enabled = false;
      let restarting = false;
      let lastWakeAt = 0;
      let lastTranscript = '';

      // Streamlit renders this widget inside an iframe. The dashboard is same-origin,
      // so we can click its real launcher and fill its real chat form.
      function parentDocument() {
        try { return window.parent.document; } catch (e) { return null; }
      }

      function sendCommand(command) {
        const doc = parentDocument();
        if (!doc) { status.textContent = 'Could not reach the dashboard. Refresh and try again.'; return; }
        const launcher = doc.querySelector('.st-key-jarvis_open_button button');
        const popupIsOpen = !!doc.querySelector('.st-key-jarvis_chat_form input[placeholder="Ask JARVIS anything..."]');
        if (!popupIsOpen && launcher) launcher.click();
        if (!command) {
          status.textContent = 'JARVIS opened. Say “JARVIS” followed by your question.';
          return;
        }
        status.textContent = 'Heard: ' + command + ' — sending to JARVIS…';
        // Wait for Streamlit to rerender the popup after the launcher click.
        setTimeout(() => {
          const d = parentDocument();
          if (!d) return;
          const input = d.querySelector('.st-key-jarvis_chat_form input[placeholder="Ask JARVIS anything..."]')
                     || d.querySelector('input[placeholder="Ask JARVIS anything..."]');
          if (!input) {
            status.textContent = 'Popup opened, but chat input was not found. Click the robot and try again.';
            return;
          }
          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
          if (setter) setter.call(input, command); else input.value = command;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
          setTimeout(() => {
            const send = d.querySelector('.st-key-jarvis_chat_form button[type="submit"]')
                      || Array.from(d.querySelectorAll('button')).find(b => b.innerText.trim().toLowerCase() === 'send');
            if (send) { send.click(); status.textContent = 'Command sent to JARVIS.'; }
            else { status.textContent = 'Your command was heard. Click Send in the JARVIS popup.'; }
          }, 250);
        }, popupIsOpen ? 100 : 900);
      }

      function startRecognition() {
        if (!SpeechRecognition) {
          status.textContent = 'Speech recognition is not supported here. Try Chrome or Edge.';
          button.textContent = 'Browser not supported'; button.disabled = true; return;
        }
        recognition = new SpeechRecognition();
        recognition.lang = 'en-IN';
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.onresult = (event) => {
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (!event.results[i].isFinal) continue;
            const transcript = (event.results[i][0].transcript || '').trim();
            if (!transcript || transcript === lastTranscript) continue;
            lastTranscript = transcript;
            const match = transcript.match(/\b(?:hey|hi|okay|ok)?\s*jarvis\b[\s,.:;-]*(.*)$/i);
            if (match && Date.now() - lastWakeAt > 3500) {
              lastWakeAt = Date.now();
              const command = (match[1] || '').trim();
              sendCommand(command);
            }
          }
        };
        recognition.onerror = (event) => {
          if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
            enabled = false;
            button.textContent = '🎙️ Enable Hey JARVIS';
            status.textContent = 'Microphone permission denied. Allow it in browser settings.';
          } else if (enabled) status.textContent = 'Listening paused; reconnecting…';
        };
        recognition.onend = () => {
          if (enabled && !restarting) {
            restarting = true;
            setTimeout(() => {
              restarting = false;
              if (enabled && recognition) { try { recognition.start(); } catch (e) {} }
            }, 700);
          }
        };
        try {
          recognition.start(); enabled = true;
          button.textContent = '🔵 Hey JARVIS is listening';
          status.textContent = 'Listening. Say “Hey JARVIS, show my tasks”.';
        } catch (e) { status.textContent = 'Could not start listening. Click the button again.'; }
      }

      button.addEventListener('click', () => {
        if (enabled) {
          enabled = false; try { if (recognition) recognition.stop(); } catch (e) {}
          button.textContent = '🎙️ Enable Hey JARVIS'; status.textContent = 'Voice assistant is off.';
        } else startRecognition();
      });

      // Keep the listening control floating above the robot launcher.
      try {
        const frame = window.frameElement;
        if (frame) {
          frame.style.position = 'fixed'; frame.style.right = '22px'; frame.style.bottom = '112px';
          frame.style.width = '250px'; frame.style.height = '72px'; frame.style.border = '0';
          frame.style.zIndex = '100000'; frame.style.background = 'transparent';
        }
      } catch (e) {}
    })();
    </script>
    """,
    height=72,
    scrolling=False,
)

# ============================================================
# 13. PRIVACY INFORMATION
# ============================================================

st.divider()

st.caption(
    "Privacy: the monitor records foreground process names and "
    "active/idle durations locally. It does not capture keystrokes, "
    "screenshots, window titles, URLs, or private document contents. "
    "When you request AI analysis, summarized routine records are "
    "sent to Groq."
)
