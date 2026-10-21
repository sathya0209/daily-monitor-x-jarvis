
import asyncio
import os
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
from scipy.signal import butter, sosfiltfilt

from jarvis_core import (
    ask_jarvis,
    get_today_summary,
    prepare_speech_text,
    transcribe_audio,
)

SAMPLE_RATE = 16000


class DesktopJarvis:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS | Personal AI")
        self.root.geometry("1120x760")
        self.root.minsize(760, 600)
        self.root.configure(bg="#080d18")

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

        # Palette: midnight navy, electric cyan, violet, and mint.
        self.bg = "#080d18"
        self.panel = "#111a2b"
        self.panel_light = "#17243a"
        self.text_color = "#edf5ff"
        self.muted = "#8fa4c2"
        self.cyan = "#46d9ff"
        self.violet = "#9b8cff"
        self.mint = "#50e3b1"

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Jarvis.TCombobox",
            fieldbackground=self.panel_light,
            background=self.panel_light,
            foreground=self.text_color,
            arrowcolor=self.cyan,
            bordercolor="#273853",
            lightcolor="#273853",
            darkcolor="#273853",
            padding=6,
        )
        style.map(
            "Jarvis.TCombobox",
            fieldbackground=[("readonly", self.panel_light)],
            foreground=[("readonly", self.text_color)],
            selectbackground=[("readonly", self.panel_light)],
            selectforeground=[("readonly", self.text_color)],
        )

        # Main page with a little breathing room around each surface.
        shell = tk.Frame(root, bg=self.bg)
        shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        # Header / brand area.
        header = tk.Frame(shell, bg=self.bg)
        header.pack(fill=tk.X, pady=(0, 14))

        brand = tk.Frame(header, bg=self.bg)
        brand.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mark = tk.Canvas(
            brand, width=54, height=54, bg=self.bg, highlightthickness=0
        )
        mark.pack(side=tk.LEFT, padx=(0, 12))
        mark.create_oval(4, 4, 50, 50, fill="#102b42", outline=self.cyan, width=2)
        mark.create_oval(12, 12, 42, 42, fill="#172444", outline=self.violet, width=1)
        mark.create_oval(19, 19, 35, 35, fill=self.cyan, outline="")
        mark.create_line(27, 0, 27, 7, fill=self.cyan, width=2)
        mark.create_line(27, 47, 27, 54, fill=self.cyan, width=2)

        title_stack = tk.Frame(brand, bg=self.bg)
        title_stack.pack(side=tk.LEFT, fill=tk.Y, pady=(2, 0))
        tk.Label(
            title_stack, text="J.A.R.V.I.S.", font=("Segoe UI", 23, "bold"),
            fg=self.text_color, bg=self.bg
        ).pack(anchor="w")
        tk.Label(
            title_stack, text="PERSONAL AI  /  VOICE COMPANION",
            font=("Segoe UI", 9, "bold"), fg=self.cyan, bg=self.bg
        ).pack(anchor="w", pady=(1, 0))

        self.status = tk.StringVar(value="SYSTEM READY")
        status_card = tk.Frame(
            header, bg="#102438", highlightbackground="#24506a",
            highlightthickness=1, padx=12, pady=9
        )
        status_card.pack(side=tk.RIGHT, anchor="center")
        self.status_dot = tk.Canvas(
            status_card, width=10, height=10, bg="#102438", highlightthickness=0
        )
        self.status_dot.pack(side=tk.LEFT, padx=(0, 7))
        self.status_dot.create_oval(1, 1, 9, 9, fill=self.mint, outline="")
        tk.Label(
            status_card, textvariable=self.status, font=("Segoe UI", 9, "bold"),
            fg=self.mint, bg="#102438"
        ).pack(side=tk.LEFT)

        # Settings strip.
        settings = tk.Frame(
            shell, bg=self.panel, highlightbackground="#263651", highlightthickness=1
        )
        settings.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            settings, text="LANGUAGE", font=("Segoe UI", 9, "bold"),
            fg=self.muted, bg=self.panel
        ).pack(side=tk.LEFT, padx=(14, 9), pady=11)
        self.language = tk.StringVar(value="English")
        ttk.Combobox(
            settings, textvariable=self.language, values=("English", "Tanglish"),
            state="readonly", width=13, style="Jarvis.TCombobox"
        ).pack(side=tk.LEFT, pady=7)
        tk.Label(
            settings, text="  •  Answers are spoken automatically",
            font=("Segoe UI", 9), fg=self.muted, bg=self.panel
        ).pack(side=tk.LEFT, padx=12)

        # Conversation surface.
        chat_card = tk.Frame(
            shell, bg=self.panel, highlightbackground="#263651", highlightthickness=1
        )
        chat_card.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        chat_top = tk.Frame(chat_card, bg=self.panel)
        chat_top.pack(fill=tk.X, padx=16, pady=(12, 6))
        tk.Label(
            chat_top, text="CONVERSATION", font=("Segoe UI", 10, "bold"),
            fg=self.text_color, bg=self.panel
        ).pack(side=tk.LEFT)
        tk.Label(
            chat_top, text="LIVE SESSION", font=("Segoe UI", 8, "bold"),
            fg=self.mint, bg=self.panel
        ).pack(side=tk.RIGHT)

        separator = tk.Frame(chat_card, bg="#263651", height=1)
        separator.pack(fill=tk.X, padx=14, pady=(0, 8))

        self.chat = scrolledtext.ScrolledText(
            chat_card, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#0d1422", fg=self.text_color, insertbackground=self.cyan,
            selectbackground="#244665", selectforeground="#ffffff",
            state=tk.DISABLED, relief=tk.FLAT, borderwidth=0,
            padx=16, pady=14, spacing1=3, spacing3=8,
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.chat.tag_configure("speaker_user", foreground=self.violet, font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("speaker_jarvis", foreground=self.cyan, font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("message", foreground=self.text_color, font=("Segoe UI", 11))
        self.chat.tag_configure("time", foreground="#617894", font=("Segoe UI", 8))

        # Message composer.
        composer = tk.Frame(
            shell, bg=self.panel, highlightbackground="#263651", highlightthickness=1
        )
        composer.pack(fill=tk.X, pady=(12, 0), side=tk.BOTTOM)
        tk.Label(
            composer, text="MESSAGE JARVIS", font=("Segoe UI", 9, "bold"),
            fg=self.muted, bg=self.panel
        ).pack(anchor="w", padx=13, pady=(10, 5))
        input_row = tk.Frame(composer, bg=self.panel)
        input_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.entry = tk.Entry(
            input_row, font=("Segoe UI", 12), bg="#0b1321", fg=self.text_color,
            insertbackground=self.cyan, relief=tk.FLAT, bd=0,
            highlightthickness=1, highlightbackground="#2a3c59",
            highlightcolor=self.cyan,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=11, padx=(0, 8))
        self.entry.bind("<Return>", lambda event: self.send_message())
        self.entry.bind("<FocusIn>", lambda event: self.entry.configure(highlightbackground=self.cyan))
        self.entry.bind("<FocusOut>", lambda event: self.entry.configure(highlightbackground="#2a3c59"))
        self.entry.focus_set()

        self.send_button = self._make_button(
            input_row, "➤  SEND", self.send_message, "#087fa8", "#25c8f4"
        )
        self.send_button.pack(side=tk.RIGHT)
        self.entry.bind("<Control-Return>", lambda event: self.send_message())

        # Action bar with differentiated visual hierarchy.
        actions = tk.Frame(shell, bg=self.bg)
        actions.pack(fill=tk.X, pady=(12, 0), side=tk.BOTTOM)
        self.mic_button = self._make_button(
            actions, "●  START RECORDING", self.toggle_recording, "#104f4b", "#1e9d8c"
        )
        self.mic_button.pack(side=tk.LEFT, padx=(0, 7))
        self.wake_button = self._make_button(
            actions, "◉  ENABLE HEY JARVIS", self.toggle_wake_word, "#29264e", "#7466e8"
        )
        self.wake_button.pack(side=tk.LEFT, padx=(0, 7))
        self._make_button(
            actions, "▤  TODAY'S SUMMARY", self.show_summary, "#1a2940", "#314968"
        ).pack(side=tk.LEFT, padx=(0, 7))
        tk.Label(
            actions, text="Made for hands-free productivity",
            font=("Segoe UI", 9), fg="#637995", bg=self.bg
        ).pack(side=tk.RIGHT, padx=4)

        self.add_message(
            "JARVIS",
            "Hey! I'm ready. Ask me anything, or enable Hey JARVIS for hands-free conversation.",
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _make_button(self, parent, text, command, background, hover):
        button = tk.Button(
            parent, text=text, command=command, bg=background, fg="#f5f9ff",
            activebackground=hover, activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), padx=13, pady=10,
            relief=tk.FLAT, bd=0, cursor="hand2",
            highlightthickness=1, highlightbackground=hover,
        )
        button.bind("<Enter>", lambda event, b=button, c=hover: b.configure(bg=c))
        button.bind("<Leave>", lambda event, b=button, c=background: b.configure(bg=c))
        return button

    def add_message(self, speaker, message):
        self.chat.configure(state=tk.NORMAL)
        stamp = datetime.now().strftime("%H:%M")
        if speaker.lower() == "you":
            tag = "speaker_user"
            label = "YOU"
        else:
            tag = "speaker_jarvis"
            label = "JARVIS"
        self.chat.insert(tk.END, f"{label}  ", tag)
        self.chat.insert(tk.END, f"{stamp}\\n", "time")
        self.chat.insert(tk.END, f"{message}\\n\\n", "message")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def send_message(self):
        try:
            message = self.entry.get().strip()
            if not message:
                self.status.set("Type a message first.")
                self.entry.focus_set()
                return

            self.entry.delete(0, tk.END)
            self.add_message("You", message)
            self.status.set("JARVIS is thinking...")
            self.root.update_idletasks()

            worker = threading.Thread(
                target=self.get_response,
                args=(message, self.language.get()),
                daemon=True,
                name="JARVIS-response",
            )
            worker.start()
        except Exception as exc:
            print(f"[JARVIS Send button error] {type(exc).__name__}: {exc}")
            self.status.set(f"Send error: {type(exc).__name__}")
            self.add_message("JARVIS", f"I couldn't start that request: {type(exc).__name__}. Check PowerShell.")

    def get_response(self, message, language):
        self.is_thinking = True
        try:
            local_answer = self.handle_local_command(message, language)
            answer = local_answer if local_answer is not None else ask_jarvis(message, language)
            if not self.closing:
                self.root.after(0, self.display_answer, answer)
        except Exception as exc:
            # Surface failures instead of making the Send button appear unresponsive.
            print(f"[JARVIS response error] {type(exc).__name__}: {exc}")
            safe_message = (
                "I hit a problem while preparing that answer. Please check the terminal for details."
            )
            if not self.closing:
                self.root.after(0, self.display_answer, safe_message)
                self.root.after(0, self.status.set, "Response error — check terminal")
        finally:
            self.is_thinking = False

    def handle_local_command(self, message, language):
        """Small set of deterministic desktop actions; return None for normal AI questions."""
        text = message.strip().lower()
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
        self.status.set("Speaking answer...")
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
        self.wake_thread = threading.Thread(
            target=self.wake_word_loop,
            daemon=True,
        )
        self.wake_thread.start()

    def wake_word_loop(self):
        """
        Wait for Hey JARVIS once, then stay in a hands-free conversation until
        the user says a clear exit phrase. Uses voice activity detection to
        capture each utterance rather than recording a fixed-length block.
        """
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        sample_rate = SAMPLE_RATE
        wake_terms = ("jarvis", "jervis", "hey jarvis", "hey jervis")
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

        def listen_for_phrase(max_seconds=12.0, silence_seconds=1.0):
            """Record one natural spoken phrase, stopping after trailing silence."""
            frame_seconds = 0.20
            frames_needed = max(1, int(silence_seconds / frame_seconds))
            max_frames = int(max_seconds / frame_seconds)
            frames = []
            speech_started = False
            quiet_frames = 0
            # RMS threshold adapts loosely to ordinary room noise.
            threshold = 0.012

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(frame_seconds * sample_rate),
            ) as stream:
                for _ in range(max_frames):
                    if self.wake_stop.is_set() or self.closing or not self.wake_enabled:
                        return None
                    data, overflowed = stream.read(int(frame_seconds * sample_rate))
                    frame = data.copy()
                    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0

                    if rms >= threshold:
                        speech_started = True
                        quiet_frames = 0
                        frames.append(frame)
                    elif speech_started:
                        frames.append(frame)
                        quiet_frames += 1
                        if quiet_frames >= frames_needed:
                            break

            if not frames or not speech_started:
                return None
            audio = np.concatenate(frames, axis=0).reshape(-1)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0:
                audio = audio / peak * 0.95
            audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            return sr.AudioData(audio_int16.tobytes(), sample_rate, 2)

        def recognize(audio_data, language_code):
            if audio_data is None:
                return ""
            try:
                return recognizer.recognize_google(
                    audio_data, language=language_code
                ).strip()
            except (sr.UnknownValueError, sr.RequestError):
                return ""

        conversation_active = False
        try:
            while self.wake_enabled and not self.wake_stop.is_set() and not self.closing:
                if not conversation_active:
                    if self.is_processing_audio or self.speech_lock.locked() or self.command_lock.locked():
                        time.sleep(0.15)
                        continue

                    # Short chunks keep wake-word detection reasonably responsive.
                    try:
                        chunk = sd.rec(
                            int(2.0 * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype="int16",
                        )
                        sd.wait()
                        if self.wake_stop.is_set() or self.closing:
                            break
                        wake_audio = sr.AudioData(chunk.tobytes(), sample_rate, 2)
                        heard = recognize(wake_audio, "en-IN").lower()
                    except Exception as exc:
                        msg = str(exc)
                        self.root.after(0, lambda m=msg: self.status.set(f"Microphone/wake error: {m}"))
                        time.sleep(0.5)
                        continue

                    if not heard or not any(term in heard for term in wake_terms):
                        continue

                    now = time.monotonic()
                    if now - self.last_command_time < 1.0:
                        continue
                    self.last_command_time = now
                    conversation_active = True
                    self.root.after(0, self.on_wake_word_detected)
                    self.root.after(0, lambda: self.status.set(
                        "Conversation started. Speak naturally; say 'bye JARVIS' to finish."
                    ))

                    # If the wake phrase also included a request, handle that request now.
                    command = re.sub(
                        r".*?\\b(?:hey\\s+)?(?:jarvis|jervis)\\b[,.!?;:\\s]*",
                        "", heard, count=1, flags=re.I
                    ).strip()
                    if command:
                        self.root.after(0, self.submit_voice_command, command)
                        while not self.closing and self.wake_enabled and (
                            self.is_thinking or self.speech_lock.locked()
                        ):
                            time.sleep(0.15)
                        # Allow the speech thread to start before checking its lock.
                        time.sleep(0.25)
                        while not self.closing and self.wake_enabled and (
                            self.is_thinking or self.speech_lock.locked()
                        ):
                            time.sleep(0.15)
                    else:
                        self.root.after(0, lambda: self.status.set(
                            "I'm listening. You can talk normally; no need to say Hey JARVIS again."
                        ))
                    continue

                # In conversation mode, listen for the next utterance without a wake word.
                if self.is_thinking or self.speech_lock.locked() or self.command_lock.locked():
                    time.sleep(0.15)
                    continue

                self.root.after(0, lambda: self.status.set(
                    "Listening to the conversation... say 'bye JARVIS' to finish."
                ))
                audio_data = listen_for_phrase(max_seconds=14.0, silence_seconds=0.9)
                if audio_data is None:
                    continue

                language_code = "ta-IN" if self.language.get() == "Tanglish" else "en-IN"
                command = recognize(audio_data, language_code)
                if not command:
                    # Try English recognition too for mixed-language phrases.
                    if language_code != "en-IN":
                        command = recognize(audio_data, "en-IN")
                    if not command:
                        continue

                normalized = re.sub(r"[^a-zA-Z0-9' ]", "", command.lower()).strip()
                is_closing = (
                    normalized in exit_phrases
                    or normalized.startswith((
                        "okay bye", "ok bye", "alright bye",
                        "thanks thats all", "thank you thats all",
                        "thats enough", "thats it", "talk to you later",
                        "i have to go", "got to go"
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
                    # Wait for goodbye speech before returning to wake-word mode.
                    while not self.closing and self.wake_enabled and self.speech_lock.locked():
                        time.sleep(0.15)
                    time.sleep(0.3)
                    continue

                self.response_complete.clear()
                self.root.after(0, self.submit_voice_command, command)
                # Keep the session open after every response; no repeated wake word.
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
            if not self.closing:
                self.root.after(0, lambda m=msg: self.status.set(f"Continuous listening error: {m}"))
        finally:
            if not self.closing:
                self.root.after(0, lambda: self.status.set(
                    'Waiting for "Hey JARVIS"...' if self.wake_enabled else "Wake-word listening stopped."
                ))

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
