
import asyncio
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import scrolledtext, ttk

import edge_tts
import noisereduce as nr
import numpy as np
import pygame
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
from scipy.signal import butter, sosfiltfilt, resample_poly
from openwakeword.model import Model as WakeWordModel

from jarvis_core import (
    ask_jarvis,
    get_today_summary,
    prepare_speech_text,
    transcribe_audio,
)

from src.monitoring.screen_time import run_monitor

SAMPLE_RATE = 16000


class DesktopJarvis:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS - Personal AI Assistant")
        self.root.geometry("1000x740")
        self.root.minsize(680, 520)
        self.root.configure(bg="#10141c")

        self.last_answer = ""
        self.closing = False
        self.is_recording = False
        self.is_processing_audio = False
        self.is_thinking = False

        self.audio_chunks = []
        self.audio_stream = None
        self.audio_lock = threading.Lock()
        self.speech_lock = threading.Lock()
        self.wake_enabled = False
        self.wake_stop = threading.Event()
        self.wake_thread = None
        self.command_lock = threading.Lock()
        self.response_complete = threading.Event()
        self.response_complete.set()
        self.last_command_time = 0.0
        self.wake_restart_attempts = 0

        tk.Label(
            root,
            text="J.A.R.V.I.S.",
            font=("Segoe UI", 28, "bold"),
            fg="#58c7ff",
            bg="#10141c",
        ).pack(pady=(12, 2))

        tk.Label(
            root,
            text="Your Personal AI Assistant",
            font=("Segoe UI", 11),
            fg="#b8c7d9",
            bg="#10141c",
        ).pack(pady=(0, 8))

        language_frame = tk.Frame(root, bg="#10141c")
        language_frame.pack(fill=tk.X, padx=18, pady=5)

        tk.Label(
            language_frame,
            text="Response language:",
            bg="#10141c",
            fg="white",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.language = tk.StringVar(value="English")
        ttk.Combobox(
            language_frame,
            textvariable=self.language,
            values=("English", "Tanglish"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT)

        self.chat = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#171e29",
            fg="#f0f4fa",
            insertbackground="white",
            state=tk.DISABLED,
            padx=12,
            pady=12,
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=18, pady=8)

        self.status = tk.StringVar(value="Ready")
        tk.Label(
            root,
            textvariable=self.status,
            bg="#10141c",
            fg="#58c7ff",
            anchor="w",
        ).pack(fill=tk.X, padx=20, pady=4)

        self.entry = tk.Entry(
            root,
            font=("Segoe UI", 12),
            bg="#202a38",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT,
        )
        self.entry.pack(fill=tk.X, padx=18, pady=8)
        self.entry.bind("<Return>", lambda event: self.send_message())

        buttons = tk.Frame(root, bg="#10141c")
        buttons.pack(fill=tk.X, padx=18, pady=(0, 14))

        tk.Button(
            buttons,
            text="Send",
            command=self.send_message,
            bg="#087fb5",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            buttons,
            text="Today's Summary",
            command=self.show_summary,
            bg="#27384b",
            fg="white",
            padx=10,
        ).pack(side=tk.LEFT, padx=4)

        # No separate text-to-speech button: every answer is spoken automatically.
        self.mic_button = tk.Button(
            buttons,
            text="Start Recording",
            command=self.toggle_recording,
            bg="#087f75",
            fg="white",
            padx=12,
        )
        self.mic_button.pack(side=tk.LEFT, padx=4)

        self.wake_button = tk.Button(
            buttons,
            text="Enable Hey JARVIS",
            command=self.toggle_wake_word,
            bg="#5b4bb7",
            fg="white",
            padx=12,
        )
        self.wake_button.pack(side=tk.LEFT, padx=4)

        self.add_message(
            "JARVIS",
            "Hello! I'm ready. Type a message or start recording. "
            "I'll speak my answers automatically. Enable Hey JARVIS for hands-free commands.",
        )

        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Start activity tracking and hands-free wake listening automatically.
        # The existing UI is preserved; no extra controls or windows are added.
        self.activity_stop = threading.Event()
        self.activity_thread = None
        self.root.after(250, self.start_activity_monitor)
        self.root.after(1200, self.startup_greeting_and_wake_word)

    def startup_greeting_and_wake_word(self):
        """Speak the startup greeting first, then begin wake-word listening."""
        if self.closing:
            return

        greeting = (
            "Hello! I'm JARVIS. Your personal assistant is ready. "
            "You can call me anytime by saying Hey JARVIS."
        )
        self.status.set("Starting JARVIS voice assistant...")

        def greet_then_listen():
            # Do not listen while the computer speakers are playing the greeting;
            # otherwise the microphone can recognize JARVIS's own voice.
            self.speak(greeting, self.language.get())
            if not self.closing:
                self.root.after(0, self.enable_wake_after_greeting)

        threading.Thread(
            target=greet_then_listen,
            name="JARVISStartupGreeting",
            daemon=True,
        ).start()

    def enable_wake_after_greeting(self):
        """Start wake listening automatically after the greeting finishes."""
        if self.closing:
            return
        if not self.wake_enabled:
            self.toggle_wake_word()
        else:
            self.status.set('Ready. Say "Hey JARVIS" whenever you need me.')

    def restart_wake_listener(self):
        """Recover from transient microphone or recognition-loop failures."""
        if self.closing or not self.wake_enabled or self.wake_stop.is_set():
            return
        if self.wake_thread is not None and self.wake_thread.is_alive():
            # The old thread may still be finishing its exception handler.
            self.root.after(750, self.restart_wake_listener)
            return

        self.wake_restart_attempts += 1
        if self.wake_restart_attempts > 5:
            self.wake_enabled = False
            self.wake_stop.set()
            self.wake_button.configure(text="Enable Hey JARVIS", bg="#5b4bb7")
            self.status.set("Wake listening stopped after repeated errors. Check microphone and internet.")
            print("[JARVIS] Wake listener stopped after 5 restart attempts.")
            return

        print(f"[JARVIS] Restarting wake listener (attempt {self.wake_restart_attempts}/5).")
        self.status.set("Recovering wake-word listening...")
        self.wake_thread = threading.Thread(
            target=self.wake_word_loop,
            name="JARVISWakeListener",
            daemon=True,
        )
        self.wake_thread.start()

    def start_activity_monitor(self):
        """Start the screen-time monitor once, in a background thread."""
        if self.closing or (
            self.activity_thread is not None
            and self.activity_thread.is_alive()
        ):
            return

        self.activity_stop.clear()

        def monitor_worker():
            try:
                run_monitor(self.activity_stop)
            except Exception as exc:
                print(
                    f"[JARVIS activity monitor] "
                    f"{type(exc).__name__}: {exc}"
                )
                if not self.closing:
                    self.root.after(
                        0,
                        lambda message=str(exc): self.status.set(
                            f"Activity monitor error: {message}"
                        ),
                    )

        self.activity_thread = threading.Thread(
            target=monitor_worker,
            name="JARVISActivityMonitor",
            daemon=True,
        )
        self.activity_thread.start()
        print("[JARVIS] Activity monitor started automatically.")

    def add_message(self, speaker, message):
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{speaker}: {message}\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def send_message(self):
        message = self.entry.get().strip()
        if not message:
            return

        self.entry.delete(0, tk.END)
        self.add_message("You", message)
        self.status.set("JARVIS is thinking...")

        threading.Thread(
            target=self.get_response,
            args=(message, self.language.get()),
            daemon=True,
        ).start()

    def get_response(self, message, language):
        self.is_thinking = True
        try:
            # Fast local actions work even if the AI service is slow.
            local_answer = self.handle_local_command(message, language)
            answer = local_answer if local_answer is not None else ask_jarvis(message, language)
            if not self.closing:
                self.root.after(0, self.display_answer, answer)
        finally:
            self.is_thinking = False

    def handle_local_command(self, message, language):
        """Small set of deterministic desktop actions; return None for normal AI questions."""
        text = message.strip().lower()
        # Speech recognition can include the wake phrase and harmless filler words.
        # Normalize only for allowlisted local commands; never execute arbitrary text.
        text = re.sub(r"^(?:hey\s+)?jarvis[\s,.:;!?-]*", "", text, flags=re.I)
        text = re.sub(r"\bplease\b", "", text).strip()
        text = re.sub(r"\s+(?:tool|app|application)$", "", text).strip()
        tanglish = language == "Tanglish"

        if any(phrase in text for phrase in ("what time", "current time", "tell me the time", "time now", "time enna", "mani enna")):
            now = datetime.now().strftime("%I:%M %p")
            return f"Ippo time {now}." if tanglish else f"The time is {now}."
        if any(phrase in text for phrase in ("what date", "today's date", "todays date", "what day is it", "date today", "innaiku enna date")):
            today = datetime.now().strftime("%A, %d %B %Y")
            return f"Innaiku {today}." if tanglish else f"Today is {today}."
        if any(phrase in text for phrase in ("open youtube", "youtube open")):
            webbrowser.open("https://www.youtube.com")
            return "YouTube open panniten." if tanglish else "Opening YouTube."
        if any(phrase in text for phrase in ("open google", "google open", "open browser", "open web browser")):
            webbrowser.open("https://www.google.com")
            return "Google open panniten." if tanglish else "Opening Google."
        if any(phrase in text for phrase in ("open gmail", "gmail open")):
            webbrowser.open("https://mail.google.com")
            return "Gmail open panniten." if tanglish else "Opening Gmail."
        if any(phrase in text for phrase in ("open chatgpt", "chatgpt open")):
            webbrowser.open("https://chatgpt.com")
            return "ChatGPT open panniten." if tanglish else "Opening ChatGPT."
        if any(phrase in text for phrase in ("open notepad", "start notepad")):
            try:
                subprocess.Popen(["notepad.exe"])
                return "Notepad open panniten." if tanglish else "Opening Notepad."
            except Exception:
                return "Notepad open panna mudiyala." if tanglish else "I couldn't open Notepad."
        if any(phrase in text for phrase in ("open calculator", "start calculator", "calculator open")):
            try:
                subprocess.Popen(["calc.exe"])
                return "Calculator open panniten." if tanglish else "Opening Calculator."
            except Exception:
                return "Calculator open panna mudiyala." if tanglish else "I couldn't open Calculator."
        if text.startswith("search web for ") or text.startswith("google search for ") or text.startswith("search google for "):
            query = re.sub(r"^(search web for |google search for |search google for )", "", message.strip(), flags=re.I).strip()
            if query:
                webbrowser.open("https://www.google.com/search?q=" + __import__("urllib.parse", fromlist=["quote_plus"]).quote_plus(query))
                return (f"Google-la {query} search panniten." if tanglish else f"Searching Google for {query}.")
        # Safe, allowlisted desktop actions only. Never execute arbitrary shell text.
        home = os.path.expanduser("~")
        safe_folders = {
            "open downloads": os.path.join(home, "Downloads"),
            "open my downloads": os.path.join(home, "Downloads"),
            "downloads open": os.path.join(home, "Downloads"),
            "open documents": os.path.join(home, "Documents"),
            "open my documents": os.path.join(home, "Documents"),
            "open desktop folder": os.path.join(home, "Desktop"),
            "open pictures": os.path.join(home, "Pictures"),
            "open my pictures": os.path.join(home, "Pictures"),
        }
        if text in safe_folders:
            folder = safe_folders[text]
            if os.path.isdir(folder):
                try:
                    subprocess.Popen(["explorer.exe", folder])
                    label = os.path.basename(folder) or "folder"
                    return f"{label} open panniten." if tanglish else f"Opening {label}."
                except Exception as exc:
                    print(f"[JARVIS automation] Could not open folder: {exc}")
                    return "Folder open panna mudiyala." if tanglish else "I couldn't open that folder."
            return "Andha folder kidaikkala." if tanglish else "That folder was not found."

        fixed_apps = {
            "open file explorer": ("explorer.exe", "File Explorer"),
            "start file explorer": ("explorer.exe", "File Explorer"),
            "open task manager": ("taskmgr.exe", "Task Manager"),
            "start task manager": ("taskmgr.exe", "Task Manager"),
            "open paint": ("mspaint.exe", "Paint"),
            "start paint": ("mspaint.exe", "Paint"),
            "open notepad": ("notepad.exe", "Notepad"),
            "start notepad": ("notepad.exe", "Notepad"),
            "open calculator": ("calc.exe", "Calculator"),
            "start calculator": ("calc.exe", "Calculator"),
        }
        if text in fixed_apps:
            executable, label = fixed_apps[text]
            try:
                subprocess.Popen([executable])
                return f"{label} open panniten." if tanglish else f"Opening {label}."
            except Exception as exc:
                print(f"[JARVIS automation] Could not launch {label}: {exc}")
                return f"{label} open panna mudiyala." if tanglish else f"I couldn't open {label}."

        if text in {"open settings", "open windows settings", "windows settings open"}:
            try:
                os.startfile("ms-settings:")
                return "Settings open panniten." if tanglish else "Opening Windows Settings."
            except Exception as exc:
                print(f"[JARVIS automation] Could not open Settings: {exc}")
                return "Settings open panna mudiyala." if tanglish else "I couldn't open Windows Settings."

        if text in {"open vscode", "open vs code", "start vscode", "open visual studio code"}:
            try:
                subprocess.Popen(["code"])
                return "VS Code open panniten." if tanglish else "Opening Visual Studio Code."
            except Exception as exc:
                print(f"[JARVIS automation] Could not launch VS Code: {exc}")
                return "VS Code command kidaikkala. PATH-la irukka check pannunga." if tanglish else "I couldn't find the VS Code command. Check whether it is available on PATH."

        if text in {"stop listening", "disable wake word", "stop hey jarvis", "sleep jarvis"}:
            self.root.after(0, self.toggle_wake_word)
            return "Seri, wake-word listening off panren." if tanglish else "Okay, I'm turning off wake-word listening."
        return None

    def display_answer(self, answer):
        if self.closing:
            return

        self.last_answer = answer
        self.add_message("JARVIS", answer)

        # Automatically speak every answer. The wake listener pauses while speech plays,
        # then returns to waiting for the next "Hey JARVIS" phrase.
        self.status.set("Speaking...")
        threading.Thread(
            target=self.speak,
            args=(answer, self.language.get()),
            daemon=True,
        ).start()


    def toggle_wake_word(self):
        """Enable/disable a desktop wake-word listener using browser-independent audio."""
        if self.wake_enabled:
            self.wake_enabled = False
            self.wake_stop.set()
            self.wake_button.configure(text="Enable Hey JARVIS", bg="#5b4bb7")
            self.status.set("Wake-word listening stopped.")
            return

        try:
            # Check that the selected microphone can be opened before starting.
            sd.query_devices(kind="input")
        except Exception as exc:
            self.status.set(f"Microphone unavailable: {exc}")
            return

        self.wake_stop.clear()
        self.wake_enabled = True
        self.wake_button.configure(text="Disable Hey JARVIS", bg="#c0392b")
        self.status.set('Listening for "Hey JARVIS"...')
        self.wake_restart_attempts = 0
        self.wake_thread = threading.Thread(
            target=self.wake_word_loop,
            name="JARVISWakeListener",
            daemon=True,
        )
        self.wake_thread.start()

    def wake_word_loop(self):
        """
        Wake-word and hands-free conversation loop.
        Uses longer wake chunks, adaptive voice activity detection, language fallbacks,
        and explicit diagnostics instead of silently swallowing recognition errors.
        """
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 250
        recognizer.pause_threshold = 0.7
        sample_rate = SAMPLE_RATE
        wake_terms = ("jarvis", "jervis")
        exit_phrases = (
            "bye", "bye jarvis", "goodbye", "goodbye jarvis",
            "that's all", "thats all", "that's enough", "thats enough",
            "that's it", "thats it", "okay thanks", "ok thanks",
            "thanks that's all", "thanks thats all", "talk to you later",
            "see you later", "i have to go", "got to go", "i'm done",
            "im done", "we are done", "i am done", "end conversation",
            "stop conversation", "go to sleep", "sleep jarvis",
            "exit conversation",
        )

        def audio_data_from_float(audio):
            """Convert captured float audio into SpeechRecognition's AudioData."""
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return None
            peak = float(np.max(np.abs(audio)))
            if peak < 0.0005:
                return None
            # Normalize gently only when the microphone level is low.
            if peak < 0.15:
                audio = audio * (0.85 / peak)
            audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            return sr.AudioData(audio_int16.tobytes(), sample_rate, 2)

        def recognize(audio_data, language_codes):
            """Try selected language first, then common fallbacks; log API failures."""
            if audio_data is None:
                return ""
            for language_code in language_codes:
                try:
                    heard = recognizer.recognize_google(
                        audio_data,
                        language=language_code,
                    ).strip()
                    if heard:
                        print(f"[JARVIS voice recognition:{language_code}] {heard}")
                        return heard
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as exc:
                    print(f"[JARVIS Google speech recognition:{language_code}] {exc}")
                    # If Google is unreachable, other language attempts won't help.
                    break
                except Exception as exc:
                    print(f"[JARVIS speech recognition:{language_code}] {type(exc).__name__}: {exc}")
                    continue
            return ""

        def listen_for_phrase(max_seconds=12.0, silence_seconds=0.85):
            """
            Capture speech with a short pre-roll and adaptive threshold.
            This avoids clipping the first word and works better with quiet microphones.
            """
            frame_seconds = 0.10
            frame_samples = int(frame_seconds * sample_rate)
            frames_needed = max(1, int(silence_seconds / frame_seconds))
            max_frames = int(max_seconds / frame_seconds)
            frames = []
            pre_roll = []
            speech_started = False
            quiet_frames = 0
            noise_samples = []

            try:
                with sd.InputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=frame_samples,
                ) as stream:
                    # Estimate room noise from the first half-second.
                    for _ in range(5):
                        if self.wake_stop.is_set() or self.closing or not self.wake_enabled:
                            return None
                        data, _ = stream.read(frame_samples)
                        frame = data.reshape(-1).copy()
                        noise_samples.append(float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0)
                        pre_roll.append(frame)

                    noise_floor = float(np.median(noise_samples)) if noise_samples else 0.001
                    threshold = max(0.0035, min(0.018, noise_floor * 2.8))
                    pre_roll_limit = 5

                    for _ in range(max_frames):
                        if self.wake_stop.is_set() or self.closing or not self.wake_enabled:
                            return None
                        data, overflowed = stream.read(frame_samples)
                        frame = data.reshape(-1).copy()
                        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0

                        if not speech_started:
                            pre_roll.append(frame)
                            if len(pre_roll) > pre_roll_limit:
                                pre_roll.pop(0)

                        if rms >= threshold:
                            if not speech_started:
                                frames.extend(pre_roll)
                                pre_roll.clear()
                            speech_started = True
                            quiet_frames = 0
                            frames.append(frame)
                        elif speech_started:
                            frames.append(frame)
                            quiet_frames += 1
                            if quiet_frames >= frames_needed:
                                break
            except Exception as exc:
                print(f"[JARVIS microphone capture] {type(exc).__name__}: {exc}")
                self.root.after(
                    0,
                    lambda m=str(exc): self.status.set(f"Microphone capture error: {m}"),
                )
                return None

            if not frames or not speech_started:
                return None
            return audio_data_from_float(np.concatenate(frames))

        # The local model is loaded once per listener thread. The model file is
        # kept in PersonalAI/models; no cloud speech recognition is used to detect
        # the wake phrase itself.
        model_dir = Path(__file__).resolve().parent / "models"
        model_files = sorted(model_dir.glob("*.onnx"))
        if not model_files:
            message = f"No Hey JARVIS ONNX model found in: {model_dir}"
            print(f"[JARVIS local wake word] {message}")
            self.root.after(0, lambda m=message: self.status.set(m))
            self.root.after(0, self.toggle_wake_word)
            return

        try:
            wake_model = WakeWordModel(
                wakeword_models=[str(model_files[0])],
                inference_framework="onnx",
            )
            print(f"[JARVIS local wake word] Loaded {model_files[0].name}")
        except Exception as exc:
            message = f"Could not load local wake-word model: {exc}"
            print(f"[JARVIS local wake word] {type(exc).__name__}: {exc}")
            self.root.after(0, lambda m=message: self.status.set(m))
            self.root.after(0, self.toggle_wake_word)
            return

        # The tested Realtek microphone defaults to 44.1 kHz. Convert its audio
        # to 16 kHz for openWakeWord, using the same 1,280-sample model frames
        # that succeeded in the standalone test.
        input_rate = 44100
        model_rate = 16000
        input_block = 4410
        model_frame_size = 1280
        wake_threshold = 0.50
        input_device = sd.default.device[0]

        def wait_for_local_wake():
            """Return True only when the local Hey JARVIS model crosses threshold."""
            audio_buffer = np.array([], dtype=np.int16)
            try:
                with sd.InputStream(
                    device=input_device,
                    samplerate=input_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=input_block,
                ) as wake_stream:
                    while (
                        self.wake_enabled
                        and not self.wake_stop.is_set()
                        and not self.closing
                    ):
                        chunk, overflowed = wake_stream.read(input_block)
                        if overflowed:
                            print("[JARVIS local wake word] Microphone buffer overflow.")
                        chunk = np.asarray(chunk, dtype=np.int16).reshape(-1)
                        converted = resample_poly(chunk, 160, 441)
                        converted = np.clip(converted, -32768, 32767).astype(np.int16)
                        audio_buffer = np.concatenate((audio_buffer, converted))

                        while len(audio_buffer) >= model_frame_size:
                            frame = audio_buffer[:model_frame_size]
                            audio_buffer = audio_buffer[model_frame_size:]
                            scores = wake_model.predict(frame)
                            score = max(
                                (float(value) for value in scores.values()),
                                default=0.0,
                            )
                            if score >= wake_threshold:
                                print(f"[JARVIS local wake word] Detected (score={score:.3f}).")
                                wake_model.reset()
                                return True
                return False
            except Exception as exc:
                msg = str(exc)
                print(f"[JARVIS local wake microphone] {type(exc).__name__}: {msg}")
                self.root.after(0, lambda m=msg: self.status.set(f"Wake microphone error: {m}"))
                time.sleep(0.5)
                return False

        conversation_active = False
        try:
            while self.wake_enabled and not self.wake_stop.is_set() and not self.closing:
                if not conversation_active:
                    if (
                        self.is_processing_audio
                        or self.is_recording
                        or self.speech_lock.locked()
                        or self.command_lock.locked()
                    ):
                        time.sleep(0.15)
                        continue

                    if not wait_for_local_wake():
                        continue
                    if self.wake_stop.is_set() or self.closing or not self.wake_enabled:
                        break

                    self.wake_restart_attempts = 0
                    now = time.monotonic()
                    if now - self.last_command_time < 1.0:
                        continue
                    self.last_command_time = now
                    conversation_active = True
                    self.root.after(0, self.on_wake_word_detected)
                    self.root.after(
                        0,
                        lambda: self.status.set(
                            "Conversation started. Speak naturally; say 'bye JARVIS' to finish."
                        ),
                    )
                    self.root.after(
                        0,
                        lambda: print("[JARVIS] Local wake detected; waiting for command speech."),
                    )
                    continue

                if (
                    self.is_thinking
                    or self.speech_lock.locked()
                    or self.command_lock.locked()
                    or self.is_recording
                    or self.is_processing_audio
                ):
                    time.sleep(0.15)
                    continue

                self.root.after(
                    0,
                    lambda: self.status.set(
                        "Listening to the conversation... say 'bye JARVIS' to finish."
                    ),
                )
                audio_data = listen_for_phrase(max_seconds=14.0, silence_seconds=0.85)
                if audio_data is None:
                    continue

                if self.language.get() == "Tanglish":
                    language_codes = ("ta-IN", "en-IN", "en-US")
                else:
                    language_codes = ("en-IN", "en-US", "ta-IN")
                command = recognize(audio_data, language_codes)
                if not command:
                    self.root.after(
                        0,
                        lambda: self.status.set(
                            "I couldn't recognize that speech. Check the microphone and internet, then try again."
                        ),
                    )
                    continue

                normalized = re.sub(r"[^a-zA-Z0-9' ]", "", command.lower()).strip()
                normalized = re.sub(r"\s+", " ", normalized)
                is_closing = (
                    normalized in exit_phrases
                    or normalized.startswith((
                        "okay bye", "ok bye", "alright bye",
                        "thanks thats all", "thank you thats all",
                        "thats enough", "thats it", "talk to you later",
                        "i have to go", "got to go",
                    ))
                )
                if is_closing:
                    conversation_active = False
                    self.root.after(0, self.add_message, "You", command)
                    goodbye = (
                        "Seri, conversation mudichikalam. Thevai na 'Hey JARVIS' nu koopidu."
                        if self.language.get() == "Tanglish"
                        else "Alright, ending our conversation. Say Hey JARVIS whenever you need me."
                    )
                    self.root.after(0, self.add_message, "JARVIS", goodbye)
                    try:
                        from jarvis_core import reset_conversation
                        reset_conversation(self.language.get())
                    except Exception:
                        pass
                    self.root.after(0, lambda: self.status.set('Waiting for "Hey JARVIS"...'))
                    threading.Thread(
                        target=self.speak,
                        args=(goodbye, self.language.get()),
                        daemon=True,
                    ).start()
                    while not self.closing and self.wake_enabled and self.speech_lock.locked():
                        time.sleep(0.15)
                    time.sleep(0.3)
                    continue

                self.response_complete.clear()
                self.root.after(0, self.submit_voice_command, command)
                while not self.closing and self.wake_enabled and (
                    self.is_thinking or self.speech_lock.locked()
                ):
                    time.sleep(0.15)
                time.sleep(0.25)
                while not self.closing and self.wake_enabled and (
                    self.is_thinking or self.speech_lock.locked()
                ):
                    time.sleep(0.15)

        except Exception as exc:
            msg = str(exc)
            print(f"[JARVIS continuous listening] {type(exc).__name__}: {msg}")
            if not self.closing:
                self.root.after(
                    0,
                    lambda m=msg: self.status.set(f"Continuous listening error: {m}"),
                )
        finally:
            if not self.closing:
                if self.wake_enabled and not self.wake_stop.is_set():
                    # Restart after transient device errors instead of leaving the
                    # UI claiming wake listening is enabled when the worker has died.
                    self.root.after(1500, self.restart_wake_listener)
                else:
                    self.root.after(
                        0,
                        lambda: self.status.set(
                            'Waiting for "Hey JARVIS"...'
                            if self.wake_enabled
                            else "Wake-word listening stopped."
                        ),
                    )

    def on_wake_word_detected(self):
        if self.closing:
            return
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(1200, lambda: self.root.attributes("-topmost", False))
        self.status.set("JARVIS activated. Listening for your command...")

    def submit_voice_command(self, command):
        if self.closing:
            return
        self.response_complete.clear()
        self.add_message("You", command)
        self.status.set("JARVIS is thinking...")
        threading.Thread(
            target=self.get_response,
            args=(command, self.language.get()),
            daemon=True,
        ).start()

    def show_summary(self):
        self.status.set("Reading your local records...")

        threading.Thread(
            target=self.get_summary,
            args=(self.language.get(),),
            daemon=True,
        ).start()

    def get_summary(self, language):
        records = get_today_summary()
        prompt = (
            "Summarize these recorded tasks and screen activity. "
            "Clearly distinguish recorded facts from assumptions. "
            "Do not invent missing information.\n\n" + records
        )
        answer = ask_jarvis(prompt, language)

        if not self.closing:
            self.root.after(0, self.display_answer, answer)

    def speak(self, text, language):
        audio_path = None
        interrupted_audio = None

        try:
            speech_text = prepare_speech_text(text, language)
            voice = (
                "ta-IN-PallaviNeural"
                if language == "Tanglish"
                else "en-IN-NeerjaNeural"
            )

            with self.speech_lock:
                if self.closing:
                    return

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
                    audio_path = audio_file.name

                async def generate_audio():
                    communicator = edge_tts.Communicate(speech_text, voice)
                    await communicator.save(audio_path)

                asyncio.run(generate_audio())
                if self.closing:
                    return

                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                # Listen for a user interruption while JARVIS is speaking.
                # A short run of loud microphone frames is treated as barge-in.
                try:
                    frame_samples = int(0.10 * SAMPLE_RATE)
                    speech_frames = []
                    consecutive_loud = 0
                    quiet_frames = 0
                    detected = False
                    threshold = 0.035

                    with sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=frame_samples,
                    ) as mic:
                        while pygame.mixer.music.get_busy() and not self.closing:
                            data, overflowed = mic.read(frame_samples)
                            frame = data.reshape(-1).copy()
                            rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0

                            if rms >= threshold:
                                consecutive_loud += 1
                                quiet_frames = 0
                                speech_frames.append(frame)
                                if consecutive_loud >= 3:
                                    detected = True
                            elif detected:
                                speech_frames.append(frame)
                                quiet_frames += 1
                                if quiet_frames >= 10:
                                    break
                            else:
                                consecutive_loud = 0
                                speech_frames.clear()

                            if detected and len(speech_frames) >= 3:
                                # Continue capturing until a brief silence after the interruption.
                                continue

                    if detected and speech_frames:
                        pygame.mixer.music.stop()
                        interrupted_audio = np.concatenate(speech_frames)
                except Exception as mic_exc:
                    # If simultaneous microphone monitoring is unavailable, keep normal playback.
                    print(f"[JARVIS barge-in] Microphone monitor unavailable: {mic_exc}")
                    while pygame.mixer.music.get_busy() and not self.closing:
                        time.sleep(0.1)

                pygame.mixer.music.unload()

            if interrupted_audio is not None and not self.closing:
                # Recognize the interruption and hand it to the same conversation flow.
                try:
                    peak = float(np.max(np.abs(interrupted_audio))) if interrupted_audio.size else 0.0
                    if peak > 0:
                        interrupted_audio = interrupted_audio / peak * 0.95
                    audio_int16 = np.int16(np.clip(interrupted_audio, -1.0, 1.0) * 32767)
                    audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)
                    recognizer = sr.Recognizer()
                    lang_code = "ta-IN" if language == "Tanglish" else "en-IN"
                    try:
                        heard = recognizer.recognize_google(audio_data, language=lang_code).strip()
                    except (sr.UnknownValueError, sr.RequestError):
                        heard = ""
                    if heard:
                        self.root.after(0, self.submit_voice_command, heard)
                    else:
                        self.root.after(0, lambda: self.status.set("I stopped speaking, but didn't catch that. Please say it again."))
                except Exception as exc:
                    print(f"[JARVIS barge-in recognition] {type(exc).__name__}: {exc}")

            if not self.closing:
                self.root.after(0, lambda: self.status.set("Listening..." if interrupted_audio is not None else "Ready"))

        except Exception as exc:
            message = str(exc)
            print(f"[JARVIS speech error] {type(exc).__name__}: {message}")
            if not self.closing:
                self.root.after(0, lambda msg=message: self.status.set(f"Speech error: {msg}"))
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            self.response_complete.set()

    def toggle_recording(self):
        if self.is_processing_audio:
            self.status.set("Please wait for speech recognition to finish.")
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        try:
            with self.audio_lock:
                self.audio_chunks = []

            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self.audio_callback,
            )
            self.audio_stream.start()

            self.is_recording = True
            self.mic_button.configure(
                text="Stop Recording",
                bg="#c0392b",
            )
            self.status.set("Listening... Speak naturally.")

        except Exception as exc:
            self.status.set(f"Microphone error: {exc}")
            self.is_recording = False

            if self.audio_stream is not None:
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None

    def audio_callback(self, indata, frames, time_info, status):
        with self.audio_lock:
            self.audio_chunks.append(indata.copy())

    def stop_recording(self):
        self.is_recording = False
        self.is_processing_audio = True

        self.mic_button.configure(
            text="Processing...",
            state=tk.DISABLED,
        )
        self.status.set("Processing your voice...")

        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            with self.audio_lock:
                chunks = list(self.audio_chunks)
                self.audio_chunks = []

            if not chunks:
                self.is_processing_audio = False
                self.reset_mic_button()
                self.status.set("No audio captured. Please try again.")
                return

            audio = np.concatenate(chunks, axis=0).reshape(-1)

            threading.Thread(
                target=self.process_recording,
                args=(audio, self.language.get()),
                daemon=True,
            ).start()

        except Exception as exc:
            self.is_processing_audio = False
            self.reset_mic_button()
            self.status.set(f"Recording error: {exc}")

    def process_recording(self, audio, language):
        audio_path = None

        try:
            self.root.after(
                0,
                lambda: self.status.set("Reducing background noise..."),
            )

            audio = audio.astype(np.float32)
            audio -= np.mean(audio)

            if len(audio) > 100:
                sos = butter(
                    4,
                    [100, 7000],
                    btype="bandpass",
                    fs=SAMPLE_RATE,
                    output="sos",
                )
                filtered = sosfiltfilt(sos, audio)
            else:
                filtered = audio

            cleaned = nr.reduce_noise(
                y=filtered,
                sr=SAMPLE_RATE,
                stationary=True,
                prop_decrease=0.80,
            )

            peak = float(np.max(np.abs(cleaned))) if len(cleaned) else 0.0
            if peak > 0:
                cleaned = cleaned / peak * 0.95

            audio_int16 = np.int16(
                np.clip(cleaned, -1.0, 1.0) * 32767
            )

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:
                audio_path = temp_file.name

            write(audio_path, SAMPLE_RATE, audio_int16)

            self.root.after(
                0,
                lambda: self.status.set("Recognizing your speech..."),
            )

            transcript = transcribe_audio(audio_path, language)

            if not transcript:
                self.root.after(
                    0,
                    lambda: self.status.set(
                        "I couldn't recognize that. Please try speaking "
                        "closer to the microphone."
                    ),
                )
            elif not self.closing:
                # Let the user correct recognition errors before AI processing.
                self.root.after(0, self.show_transcription, transcript)

        except Exception as exc:
            message = str(exc)
            if not self.closing:
                self.root.after(
                    0,
                    lambda msg=message: self.status.set(
                        f"Speech recognition error: {msg}"
                    ),
                )

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

            self.is_processing_audio = False
            if not self.closing:
                self.root.after(0, self.reset_mic_button)

    def show_transcription(self, text):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.entry.focus_set()
        self.status.set(
            "Check the recognized text, then press Enter or Send."
        )

    def reset_mic_button(self):
        if not self.closing:
            self.mic_button.configure(
                text="Start Recording",
                state=tk.NORMAL,
                bg="#087f75",
            )

    def close(self):
        self.closing = True
        self.wake_enabled = False
        self.wake_stop.set()

        # Ask the monitor to save its final segment before the app exits.
        if hasattr(self, "activity_stop"):
            self.activity_stop.set()
        if getattr(self, "activity_thread", None) is not None:
            self.activity_thread.join(timeout=2.0)

        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass

        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopJarvis(root)
    root.mainloop()
