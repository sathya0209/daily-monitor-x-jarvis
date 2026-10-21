
import asyncio
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sqlite3
import tempfile
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import scrolledtext, ttk
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import edge_tts
import noisereduce as nr
import numpy as np
import pygame
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
from scipy.signal import butter, sosfiltfilt, resample_poly
from openwakeword.model import Model as WakeWordModel
from PIL import Image, ImageTk

from jarvis_core import (
    ask_jarvis,
    get_today_summary,
    prepare_speech_text,
    transcribe_audio,
)

from src.monitoring.screen_time import run_monitor
from jarvis_shared_commands import handle_shared_desktop_command

SAMPLE_RATE = 16000


class DesktopJarvis:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS")
        self.root.overrideredirect(True)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        bar_width, bar_height, margin = 190, 260, 24
        self.root.geometry(
            f"{bar_width}x{bar_height}+{max(0, screen_width - bar_width - margin)}+"
            f"{max(0, screen_height - bar_height - margin)}"
        )
        self.always_on_top = True
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#010203")
        try:
            # Windows color-key transparency removes the rectangular canvas background.
            self.root.attributes("-transparentcolor", "#010203")
        except tk.TclError:
            pass

        # Transparent character launcher; clicking Luffy opens the chat panel.
        self.floating_bar = tk.Frame(root, bg="#010203")
        self.floating_bar.pack(fill=tk.BOTH, expand=True)
        self.robot_canvas = tk.Canvas(
            self.floating_bar, width=190, height=260, bg="#010203",
            highlightthickness=0, cursor="hand2",
        )
        self.robot_canvas.pack(fill=tk.BOTH, expand=True)
        self._robot_press = None
        self._robot_last_pointer = None
        self._robot_was_near = False
        self._robot_greet_until = 0.0
        self._robot_play_until = 0.0
        self._robot_touch_until = 0.0
        self.weather = None
        self.weather_kind = None
        self.weather_location = "Chennai"
        self.weather_error = False
        # Use the city the user confirmed; IP lookup can point at an ISP gateway.
        self._weather_coords = (13.08, 80.26)
        self._weather_reaction_until = 0.0
        self.current_gear = 1
        self._robot_gear_flash_until = 0.0
        self._gear_transition_start = 0.0
        self._gear_transition_from = 1
        self._gear_transition_to = 1
        self._is_maximized = False
        self._restore_panel_geometry = None
        self._skip_next_speech = False
        self._shutdown_after_speech = False
        self._background_hidden = False
        self.character_frames = self.load_character_frames()
        self.robot_canvas.bind("<ButtonPress-1>", self.begin_robot_press)
        self.robot_canvas.bind("<B1-Motion>", self.drag_robot)
        self.robot_canvas.bind("<ButtonRelease-1>", self.release_robot)
        self.robot_canvas.bind("<Button-3>", self.playful_reaction)

        self.panel_header = tk.Frame(root, bg="#111c2d", height=48,
                                     highlightthickness=1, highlightbackground="#2d5877")
        self.panel_header.pack_propagate(False)
        tk.Label(self.panel_header, text="*", font=("Segoe UI", 17, "bold"),
                 fg="#58c7ff", bg="#111c2d").pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(self.panel_header, text="JARVIS", font=("Segoe UI", 11, "bold"),
                 fg="#f3f7ff", bg="#111c2d").pack(side=tk.LEFT)
        self.maximize_button = tk.Button(
            self.panel_header, text="□", command=self.toggle_maximize,
            font=("Segoe UI", 11), bg="#111c2d", fg="#b8c7d9",
            activebackground="#24344a", activeforeground="white",
            relief=tk.FLAT, bd=0, width=4, cursor="hand2",
        )
        self.maximize_button.pack(side=tk.RIGHT)
        tk.Button(self.panel_header, text="-", command=self.show_bar,
                  font=("Segoe UI", 13), bg="#111c2d", fg="#b8c7d9",
                  activebackground="#24344a", activeforeground="white",
                  relief=tk.FLAT, bd=0, width=4, cursor="hand2").pack(side=tk.RIGHT)
        tk.Button(self.panel_header, text="x", command=self.hide_to_background,
                  font=("Segoe UI", 15), bg="#111c2d", fg="#b8c7d9",
                  activebackground="#9c3443", activeforeground="white",
                  relief=tk.FLAT, bd=0, width=3, cursor="hand2").pack(side=tk.RIGHT)

        self.content = tk.Frame(root, bg="#10141c")
        self._drag_origin = None
        for widget in (self.panel_header,):
            widget.bind("<ButtonPress-1>", self.begin_drag)
            widget.bind("<B1-Motion>", self.drag_window)
        for widget in self.panel_header.winfo_children():
            if not isinstance(widget, tk.Button):
                widget.bind("<ButtonPress-1>", self.begin_drag)
                widget.bind("<B1-Motion>", self.drag_window)

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
            self.content,
            text="J.A.R.V.I.S.",
            font=("Segoe UI", 28, "bold"),
            fg="#58c7ff",
            bg="#10141c",
        ).pack(pady=(12, 2))

        tk.Label(
            self.content,
            text="Your Personal AI Assistant",
            font=("Segoe UI", 11),
            fg="#b8c7d9",
            bg="#10141c",
        ).pack(pady=(0, 8))

        language_frame = tk.Frame(self.content, bg="#10141c")
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
            values=("English", "Tanglish", "Japanese"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT)

        preferences = tk.Frame(self.content, bg="#10141c")
        preferences.pack(fill=tk.X, padx=18, pady=(1, 6))
        tk.Label(preferences, text="Theme", bg="#10141c", fg="#b8c7d9").pack(side=tk.LEFT)
        self.theme = tk.StringVar(value="Ocean")
        theme_picker = ttk.Combobox(preferences, textvariable=self.theme,
                                    values=("Ocean", "Sunset", "High contrast"),
                                    state="readonly", width=13)
        theme_picker.pack(side=tk.LEFT, padx=(6, 14))
        theme_picker.bind("<<ComboboxSelected>>", self.apply_accessibility_settings)
        tk.Label(preferences, text="Text", bg="#10141c", fg="#b8c7d9").pack(side=tk.LEFT)
        self.text_size = tk.StringVar(value="Normal")
        size_picker = ttk.Combobox(preferences, textvariable=self.text_size,
                                   values=("Normal", "Large", "Extra large"),
                                   state="readonly", width=12)
        size_picker.pack(side=tk.LEFT, padx=6)
        size_picker.bind("<<ComboboxSelected>>", self.apply_accessibility_settings)
        gear_frame = tk.Frame(self.content, bg="#10141c")
        gear_frame.pack(fill=tk.X, padx=18, pady=(0, 3))
        tk.Label(gear_frame, text="Luffy gear", bg="#10141c", fg="#b8c7d9").pack(side=tk.LEFT)
        self.gear_choice = tk.StringVar(value="Gear 1")
        self.gear_picker = ttk.Combobox(
            gear_frame, textvariable=self.gear_choice,
            values=tuple(f"Gear {number}" for number in range(1, 6)),
            state="readonly", width=10,
        )
        self.gear_picker.pack(side=tk.LEFT, padx=(8, 0))
        self.gear_picker.bind("<<ComboboxSelected>>", self.select_gear)
        self.gear_badge = tk.Label(gear_frame, text="LUFFY  ·  GEAR 1",
                                   font=("Segoe UI", 9, "bold"), bg="#10141c",
                                   fg="#ffbd59")
        self.gear_badge.pack(side=tk.RIGHT, padx=4)

        self.chat = scrolledtext.ScrolledText(
            self.content,
            wrap=tk.WORD,
            height=12,
            font=("Segoe UI", 11),
            bg="#171e29",
            fg="#f0f4fa",
            insertbackground="white",
            state=tk.DISABLED,
            padx=12,
            pady=12,
        )
        self.chat.tag_configure("speaker_you", foreground="#73d7ff", font=("Segoe UI", 9, "bold"), justify="right")
        self.chat.tag_configure("speaker_jarvis", foreground="#5de1ce", font=("Segoe UI", 9, "bold"))
        self.chat.tag_configure("message_you", foreground="#f0f6ff", background="#203a54",
                                justify="right", lmargin1=55, lmargin2=55, rmargin=12, spacing1=7, spacing3=12)
        self.chat.tag_configure("message_jarvis", foreground="#e3ebf7", background="#152239",
                                lmargin1=12, lmargin2=12, rmargin=48, spacing1=7, spacing3=12)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=18, pady=8)

        self.status = tk.StringVar(value="Ready")
        tk.Label(
            self.content,
            textvariable=self.status,
            bg="#10141c",
            fg="#58c7ff",
            anchor="w",
        ).pack(fill=tk.X, padx=20, pady=4)

        self.entry = tk.Entry(
            self.content,
            font=("Segoe UI", 12),
            bg="#202a38",
            fg="#9badc2",
            insertbackground="white",
            relief=tk.FLAT,
        )
        self.entry_placeholder = "Message JARVIS..."
        self.entry.insert(0, self.entry_placeholder)
        self.entry.pack(fill=tk.X, padx=18, pady=8)
        self.entry.bind("<Return>", lambda event: self.send_message())
        self.entry.bind("<FocusIn>", self.clear_entry_placeholder)
        self.entry.bind("<FocusOut>", self.restore_entry_placeholder)

        buttons = tk.Frame(self.content, bg="#10141c")
        buttons.pack(fill=tk.X, padx=18, pady=(0, 14))
        primary_buttons = tk.Frame(buttons, bg="#10141c")
        primary_buttons.pack(fill=tk.X)
        voice_buttons = tk.Frame(buttons, bg="#10141c")
        voice_buttons.pack(fill=tk.X, pady=(5, 0))

        tk.Button(
            primary_buttons,
            text="Send",
            command=self.send_message,
            bg="#087fb5",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            primary_buttons,
            text="Today's Plan",
            command=self.show_summary,
            bg="#27384b",
            fg="white",
            padx=10,
        ).pack(side=tk.LEFT, padx=4)

        # No separate text-to-speech button: every answer is spoken automatically.
        self.mic_button = tk.Button(
            voice_buttons,
            text="Start Recording",
            command=self.toggle_recording,
            bg="#087f75",
            fg="white",
            padx=12,
        )
        self.mic_button.pack(side=tk.LEFT, padx=4)

        self.wake_button = tk.Button(
            voice_buttons,
            text="Enable Hey JARVIS",
            command=self.toggle_wake_word,
            bg="#5b4bb7",
            fg="white",
            padx=12,
        )
        self.wake_button.pack(side=tk.LEFT, padx=4)

        self.pin_button = tk.Button(
            voice_buttons,
            text="Unpin",
            command=self.toggle_always_on_top,
            bg="#27384b",
            fg="white",
            padx=10,
        )
        self.pin_button.pack(side=tk.RIGHT, padx=4)

        self.add_message(
            "JARVIS",
            "Hello! I'm ready. Type a message or say 'Hey JARVIS' to talk. "
            "Closing this panel keeps me listening in the background.",
        )

        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_background)
        self.apply_accessibility_settings()

        # Start activity tracking and hands-free wake listening automatically.
        # The floating assistant starts tracking and listening when launched.
        self.activity_stop = threading.Event()
        self.activity_thread = None
        self.root.after(250, self.start_activity_monitor)
        self.root.after(400, self.refresh_weather)
        self.root.after(1200, self.startup_greeting_and_wake_word)
        self.root.after(40, self.animate_robot)

    def load_character_frames(self):
        """Load and size the generated transparent character poses once."""
        pose_files = {
            "idle": "luffy_idle.png",
            "wave": "luffy_wave.png",
            "rain": "luffy_rain.png",
            "walk": "luffy_walk.png",
            "run": "luffy_run.png",
            "talk": "luffy_talk.png",
            "play": "luffy_play.png",
            "stretch": "luffy_stretch.png",
            "gear2": "luffy_gear2.png",
            "gear3": "luffy_gear3.png",
            "gear4": "luffy_gear4.png",
            "gear5": "luffy_gear5.png",
        }
        asset_dir = Path(__file__).resolve().parent / "assets"
        frames = {}
        for pose, filename in pose_files.items():
            image_path = asset_dir / filename
            if not image_path.is_file():
                continue
            image = Image.open(image_path).convert("RGBA")
            alpha = image.getchannel("A")
            bounds = alpha.point(lambda value: 255 if value > 18 else 0).getbbox()
            if bounds:
                left, top, right, bottom = bounds
                pad = 12
                image = image.crop((max(0, left-pad), max(0, top-pad),
                                    min(image.width, right+pad),
                                    min(image.height, bottom+pad)))
            image.thumbnail((180, 216), Image.Resampling.LANCZOS)
            frames[pose] = ImageTk.PhotoImage(image)
        if "idle" not in frames:
            raise FileNotFoundError(f"JARVIS character art is missing from {asset_dir}")
        return frames

    def playful_reaction(self, _event=None):
        """Play a cheeky character pose and spoken quip on right-click."""
        self._robot_play_until = time.monotonic() + 2.5
        self.say_character_line("ししし！宝を探しに行こう！", "Heh-heh! Let's go find treasure!")

    def say_character_line(self, japanese, english, skip_chat_speech=False):
        language = self.language.get()
        line = japanese if language == "Japanese" else english
        if language == "Tanglish":
            line = english
        self._skip_next_speech = skip_chat_speech
        threading.Thread(target=self.speak, args=(line, language), daemon=True).start()

    def touch_reaction(self):
        """React to a tap before expanding the floating character into chat."""
        self._robot_touch_until = time.monotonic() + 1.5
        if self.weather_kind in {"rain", "storm"}:
            line = "Careful, nakama! I've got the umbrella!"
        elif self.weather_kind == "sun":
            line = "Hey! It's a great day for an adventure!"
        else:
            line = "Hey! That tickles!"
        self.say_character_line(line, line)
        self.root.after(650, self.show_panel)

    def change_gear(self, gear, skip_chat_speech=False):
        now = time.monotonic()
        self._gear_transition_from = self.current_gear
        self._gear_transition_to = gear
        self._gear_transition_start = now
        self.current_gear = gear
        self._robot_gear_flash_until = now + 1.65
        if hasattr(self, "gear_choice"):
            self.gear_choice.set(f"Gear {gear}")
        if hasattr(self, "gear_badge"):
            self.gear_badge.configure(text=f"LUFFY  ·  GEAR {gear}")
        phrases = {
            1: ("よー！", "Yo! Back to Gear 1!"),
            2: ("ギア2！", "Gear 2! Here I go!"),
            3: ("ギア3！", "Gear 3! Watch this!"),
            4: ("ギア4！", "Gear 4! Boundman!"),
            5: ("ギア5！自由にやらせてもらうぞ！", "Gear 5! I'm free to do what I want!"),
        }
        japanese, english = phrases[gear]
        self.say_character_line(japanese, english, skip_chat_speech=skip_chat_speech)
        self.status.set(f"Luffy changed to Gear {gear}")

    def select_gear(self, _event=None):
        """Switch the mascot form from the visible gear picker."""
        try:
            gear = int(self.gear_choice.get().split()[-1])
        except (ValueError, IndexError):
            return
        self.change_gear(gear)

    def apply_accessibility_settings(self, _event=None):
        palettes = {
            "Ocean": {"panel":"#101b2e", "surface":"#14243a", "input":"#1c3048", "text":"#f2f7ff", "muted":"#a8bfd8", "accent":"#00b8d9", "button":"#087fb5", "button_text":"#ffffff"},
            "Sunset": {"panel":"#241528", "surface":"#321d35", "input":"#402541", "text":"#fff3f0", "muted":"#e4becb", "accent":"#ff7485", "button":"#da4767", "button_text":"#ffffff"},
            "High contrast": {"panel":"#000000", "surface":"#080808", "input":"#000000", "text":"#ffffff", "muted":"#ffffff", "accent":"#ffff00", "button":"#ffff00", "button_text":"#000000"},
        }
        palette = palettes.get(self.theme.get(), palettes["Ocean"])
        try:
            scale = {"Normal": 1.0, "Large": 1.15, "Extra large": 1.3}.get(self.text_size.get(), 1.0)
            for widget in (self.panel_header, self.content, *self.content.winfo_children()):
                self._style_widget_tree(widget, palette, scale)
            style = ttk.Style(self.root)
            style.configure("TCombobox", fieldbackground=palette["input"], background=palette["surface"],
                            foreground=palette["text"], arrowcolor=palette["accent"])
            style.map("TCombobox", fieldbackground=[("readonly", palette["input"])],
                      foreground=[("readonly", palette["text"])])
            if hasattr(self, "chat"):
                self.chat.tag_configure("speaker_you", foreground=palette["accent"])
                self.chat.tag_configure("speaker_jarvis", foreground="#5de1ce" if self.theme.get() != "High contrast" else "#ffff00")
                self.chat.tag_configure("message_you", foreground=palette["text"], background=palette["input"], justify="right")
                self.chat.tag_configure("message_jarvis", foreground=palette["text"], background=palette["surface"])
        except (tk.TclError, AttributeError):
            pass

    def _style_widget_tree(self, widget, palette, scale):
        try:
            kind = widget.winfo_class()
            if kind in ("Frame", "Labelframe"):
                widget.configure(bg=palette["panel"])
            elif kind == "Label":
                widget.configure(bg=palette["panel"], fg=palette["text"])
            elif kind == "Button":
                widget.configure(bg=palette["button"], fg=palette["button_text"],
                                 activebackground=palette["accent"], activeforeground=palette["button_text"])
            elif kind == "Entry":
                widget.configure(bg=palette["input"], fg=palette["text"], insertbackground=palette["text"])
            elif kind == "Text":
                widget.configure(bg=palette["surface"], fg=palette["text"], insertbackground=palette["text"])
            if "font" in widget.keys():
                if not hasattr(widget, "_jarvis_font_base"):
                    family = widget.tk.call("font", "actual", widget.cget("font"), "-family")
                    size = int(widget.tk.call("font", "actual", widget.cget("font"), "-size"))
                    weight = widget.tk.call("font", "actual", widget.cget("font"), "-weight")
                    widget._jarvis_font_base = (family, abs(size), weight)
                family, base_size, weight = widget._jarvis_font_base
                widget.configure(font=(family, max(9, round(base_size * scale)), weight))
        except (tk.TclError, ValueError):
            pass
        for child in widget.winfo_children():
            self._style_widget_tree(child, palette, scale)

    def refresh_weather(self):
        """Fetch approximate current-location weather without blocking the UI."""
        if self.closing:
            return
        threading.Thread(
            target=self.fetch_weather,
            name="JARVISWeather",
            daemon=True,
        ).start()
        self.root.after(20 * 60 * 1000, self.refresh_weather)

    def fetch_weather(self):
        """Fetch current weather for the user's confirmed city."""
        try:
            latitude, longitude = self._weather_coords
            query = urlencode({
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code,is_day,precipitation",
                "timezone": "auto",
            })
            request = Request(
                f"https://api.open-meteo.com/v1/forecast?{query}",
                headers={"User-Agent": "JARVIS-Desktop/1.0"},
            )
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            current = payload["current"]
            code = int(current["weather_code"])
            is_day = bool(current.get("is_day", 1))
            if code >= 95:
                kind, label = "storm", "Thunderstorm"
            elif code in (71, 73, 75, 77, 85, 86):
                kind, label = "snow", "Snow"
            elif 51 <= code <= 67 or 80 <= code <= 82:
                kind, label = "rain", "Rain"
            elif code in (45, 48):
                kind, label = "fog", "Fog"
            elif code == 0 and not is_day:
                kind, label = "night", "Clear night"
            elif code == 0:
                kind, label = "sun", "Sunny"
            elif code in (1, 2, 3):
                kind, label = "cloud", "Cloudy"
            else:
                kind, label = "cloud", "Cloudy"
            result = {
                "kind": kind,
                "label": label,
                "temperature": round(float(current["temperature_2m"])),
                "is_day": is_day,
            }
            self.root.after(0, self.apply_weather, result)
        except Exception as exc:
            print(f"[JARVIS weather] {type(exc).__name__}: {exc}")
            if not self.closing:
                self.root.after(0, self.set_weather_unavailable)

    def set_weather_unavailable(self):
        self.weather_error = True

    def apply_weather(self, weather):
        """Update the mascot's weather props and react when conditions change."""
        if self.closing:
            return
        previous = self.weather_kind
        self.weather = weather
        self.weather_kind = weather["kind"]
        self.weather_error = False
        if previous is not None and previous != self.weather_kind:
            self._weather_reaction_until = time.monotonic() + 3.0
            quips = {
                "rain": ("A little rain? Umbrella ready!", "Rain? Umbrella ready!"),
                "storm": ("Storm clouds! Stay safe, nakama!", "Storm incoming! Stay safe!"),
                "sun": ("Sun's out! Adventure time!", "Sunny skies! Adventure time!"),
                "cloud": ("Clouds rolling in. I am keeping watch!", "Cloudy skies. I'm keeping watch!"),
                "snow": ("Snow! This island looks amazing!", "Snowfall! What a sight!"),
                "fog": ("It's foggy. Stay close, nakama!", "Foggy out there. Stay close!"),
                "night": ("Night has fallen. Time to rest up!", "Clear night. Time to rest up!"),
            }
            japanese, english = quips.get(self.weather_kind, quips["cloud"])
            self.say_character_line(japanese, english)

    def animate_robot(self):
        """Choose the character pose from cursor activity and JARVIS speech."""
        if self.closing or not self.robot_canvas.winfo_exists():
            return

        now = time.monotonic()
        pointer_x, pointer_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        center_x, center_y = self.root.winfo_x() + 95, self.root.winfo_y() + 105
        distance = math.hypot(pointer_x-center_x, pointer_y-center_y)
        previous = self._robot_last_pointer
        elapsed = max(0.01, now - previous[2]) if previous else 0.055
        speed = math.hypot(pointer_x-previous[0], pointer_y-previous[1]) / elapsed if previous else 0
        self._robot_last_pointer = (pointer_x, pointer_y, now)

        near = distance < 190
        if near and not self._robot_was_near:
            self._robot_greet_until = now + 1.6
            self.say_character_line("よー！", "Yo!")
        self._robot_was_near = near

        gear_poses = {1: "idle", 2: "gear2", 3: "gear3", 4: "gear4", 5: "gear5"}
        transforming = now < self._robot_gear_flash_until
        if transforming:
            duration = max(0.01, self._robot_gear_flash_until - self._gear_transition_start)
            progress = min(1.0, max(0.0, (now - self._gear_transition_start) / duration))
            if progress < 0.2:
                pose = gear_poses[self._gear_transition_from]
            elif progress < 0.76:
                pose = "stretch" if int(now * 10) % 2 == 0 else gear_poses[self._gear_transition_from]
            else:
                pose = gear_poses[self._gear_transition_to]
        elif now < self._robot_touch_until:
            pose = "wave"
        elif now < self._robot_play_until:
            pose = "stretch" if int(now * 8) % 2 == 0 else "play"
        elif self.speech_lock.locked():
            # Alternate open-mouth and neutral art so speech feels animated.
            pose = "talk" if int(now * 4) % 2 == 0 else "idle"
        elif near and speed > 750:
            pose = "run"
        elif near and speed > 90:
            pose = "walk"
        elif now < self._weather_reaction_until:
            pose = "wave"
        elif self.weather_kind in {"rain", "storm"} and "rain" in self.character_frames:
            pose = "rain"
        elif near and now < self._robot_greet_until:
            pose = "wave"
        else:
            pose = "idle"

        if (
            not transforming
            and self.weather_kind in {"rain", "storm"}
            and "rain" in self.character_frames
        ):
            pose = "rain"
        if (
            not transforming
            and self.weather_kind not in {"rain", "storm"}
            and self.current_gear > 1
            and self.current_gear in gear_poses
        ):
            form_pose = gear_poses[self.current_gear]
            if form_pose in self.character_frames:
                pose = form_pose
        image = self.character_frames.get(pose) or self.character_frames["idle"]
        bob = math.sin(now * (14 if pose in {"walk", "run"} else 3)) * (
            3 if pose in {"walk", "run"} else 1.2
        )
        if transforming:
            progress = min(1.0, max(0.0, (now - self._gear_transition_start) / max(0.01, self._robot_gear_flash_until - self._gear_transition_start)))
            bob += math.sin(progress * math.pi * 4) * 7
        x = 95 + (math.sin(now * 12) * 3 if pose in {"walk", "run"} else 0)
        self.robot_canvas.delete("all")
        gear_colors = {1: "#ffbd59", 2: "#ff5577", 3: "#ffc75b", 4: "#ff657a", 5: "#75e9ff"}
        if transforming:
            pulse = abs(math.sin(progress * math.pi))
            radius = 30 + pulse * 76
            accent = gear_colors[self._gear_transition_to]
            self.robot_canvas.create_oval(95-radius, 112-radius, 95+radius, 112+radius,
                                          outline=accent, width=3)
            self.robot_canvas.create_oval(95-radius*0.65, 112-radius*0.65,
                                          95+radius*0.65, 112+radius*0.65,
                                          outline="#fff2cc", width=1)
        elif self.current_gear == 2:
            aura = "#ff5577" if int(now * 8) % 2 == 0 else "#ff9a72"
            self.robot_canvas.create_oval(27, 22, 163, 211, outline=aura, width=3)
        self.robot_canvas.create_oval(48, 218, 142, 226, fill="#1b2533", outline="")
        self.robot_canvas.create_image(x, 111+bob, image=image, anchor=tk.CENTER)
        self.robot_canvas.image = image
        self.draw_weather_overlay(now)
        if near and pose not in {"talk", "play", "rain"}:
            bubble_text = "ありがとう！" if pose == "wave" else f"GEAR {self.current_gear}!"
            accent = {2: "#ff637d", 3: "#ffc75b", 4: "#ff657a", 5: "#75e9ff"}.get(self.current_gear, "#ae7e36")
            self.robot_canvas.create_oval(139, 42, 185, 66, fill="#fff7dc", outline=accent, width=2)
            self.robot_canvas.create_text(162, 54, text=bubble_text,
                                          fill="#8a2e25", font=("Segoe UI", 9, "bold"))
        self.root.after(70, self.animate_robot)

    def draw_weather_overlay(self, now):
        """Draw small weather cues and a rain umbrella over the floating mascot."""
        if not self.weather:
            if self.weather_error:
                self.robot_canvas.create_rectangle(4, 236, 186, 256, fill="#101a2a", outline="#53718d", width=1)
                label = "Weather unavailable"
            else:
                label = "Weather loading..."
            self.robot_canvas.create_text(95, 246, text=label, fill="#f1f7ff",
                                         font=("Segoe UI", 8, "bold"))
            return
        canvas = self.robot_canvas
        kind = self.weather["kind"]
        location = self.weather_location
        description = f"{self.weather['label']}  {self.weather['temperature']}°"
        if location:
            description += f" · {location}"
        if len(description) > 27:
            description = description[:26].rstrip() + "…"
        canvas.create_rectangle(4, 236, 186, 256, fill="#101a2a", outline="#53718d", width=1)
        canvas.create_text(95, 246, text=description, fill="#f1f7ff",
                           font=("Segoe UI", 8, "bold"))

        if kind in {"sun", "cloud", "night"}:
            cx, cy = 27, 43
            if kind == "sun":
                for angle in range(0, 360, 45):
                    radians = math.radians(angle)
                    x1, y1 = cx + math.cos(radians) * 12, cy + math.sin(radians) * 12
                    x2, y2 = cx + math.cos(radians) * 17, cy + math.sin(radians) * 17
                    canvas.create_line(x1, y1, x2, y2, fill="#ffd54a", width=2)
                canvas.create_oval(cx-9, cy-9, cx+9, cy+9, fill="#ffd54a", outline="#fff1a8")
            elif kind == "night":
                canvas.create_oval(cx-9, cy-9, cx+9, cy+9, fill="#dce8ff", outline="")
                canvas.create_oval(cx-3, cy-12, cx+11, cy+3, fill="#010203", outline="")
            else:
                cloud = "#d7e5f4"
                canvas.create_oval(cx-12, cy-4, cx+2, cy+9, fill=cloud, outline="")
                canvas.create_oval(cx-4, cy-11, cx+11, cy+9, fill=cloud, outline="")
                canvas.create_oval(cx+4, cy-5, cx+16, cy+9, fill=cloud, outline="")
                canvas.create_rectangle(cx-7, cy+1, cx+11, cy+9, fill=cloud, outline="")

        if kind in {"rain", "storm"}:
            drop = "#76dfff" if kind == "rain" else "#b6b5ff"
            offset = int((now * 20) % 10)
            for px, py in ((15, 89), (30, 117), (175, 103), (22, 151), (168, 145)):
                canvas.create_line(px, py + offset, px - 4, py + 8 + offset,
                                   fill=drop, width=2)
        elif kind == "snow":
            for index, (px, py) in enumerate(((24, 94), (167, 111), (25, 151), (170, 158))):
                wobble = math.sin(now * 2 + index) * 3
                canvas.create_oval(px+wobble-3, py-3, px+wobble+3, py+3,
                                   fill="#aeeaff", outline="#e7fbff")
        elif kind == "fog":
            for row in range(3):
                y = 86 + row * 12
                canvas.create_line(7, y, 35, y-2, fill="#d6e0ea", width=2)
                canvas.create_line(158, y-2, 185, y, fill="#d6e0ea", width=2)

    def begin_robot_press(self, event):
        self._robot_press = (event.x_root, event.y_root,
                             self.root.winfo_x(), self.root.winfo_y(), False)

    def drag_robot(self, event):
        if self._robot_press is None:
            return
        start_x, start_y, window_x, window_y, moved = self._robot_press
        dx, dy = event.x_root - start_x, event.y_root - start_y
        moved = moved or math.hypot(dx, dy) > 6
        self._robot_press = (start_x, start_y, window_x, window_y, moved)
        if moved:
            self.root.geometry(f"+{window_x + dx}+{window_y + dy}")

    def release_robot(self, _event):
        if self._robot_press is None:
            return
        moved = self._robot_press[4]
        self._robot_press = None
        if not moved:
            self.touch_reaction()

    def show_panel(self):
        """Expand the robot launcher into the chat panel."""
        self._background_hidden = False
        self.root.deiconify()
        self.floating_bar.pack_forget()
        self.panel_header.pack(fill=tk.X)
        self.content.pack(fill=tk.BOTH, expand=True)
        width, height, margin = 460, 720, 24
        max_x = max(0, self.root.winfo_screenwidth() - width - margin)
        max_y = max(0, self.root.winfo_screenheight() - height - margin)
        x = min(max(0, self.root.winfo_x()), max_x)
        y = min(max(0, self.root.winfo_y()), max_y)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.lift()
        self.root.attributes("-topmost", self.always_on_top)
        self.entry.focus_set()

    def show_bar(self):
        """Collapse JARVIS back to its animated robot mascot."""
        self._is_maximized = False
        self._restore_panel_geometry = None
        self.maximize_button.configure(text="□")
        self.panel_header.pack_forget()
        self.content.pack_forget()
        self.floating_bar.pack(fill=tk.BOTH, expand=True)
        width, height, margin = 190, 260, 24
        max_x = max(0, self.root.winfo_screenwidth() - width - margin)
        max_y = max(0, self.root.winfo_screenheight() - height - margin)
        x = min(max(0, self.root.winfo_x()), max_x)
        y = min(max(0, self.root.winfo_y()), max_y)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.lift()
        self.root.attributes("-topmost", self.always_on_top)

    def toggle_maximize(self):
        """Toggle the chat panel between its normal and full-screen size."""
        if not self._is_maximized:
            self._restore_panel_geometry = (
                self.root.winfo_width(), self.root.winfo_height(),
                self.root.winfo_x(), self.root.winfo_y(),
            )
            self.root.geometry(
                f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
            )
            self.maximize_button.configure(text="❐")
            self._is_maximized = True
        else:
            width, height, x, y = self._restore_panel_geometry or (460, 720, 24, 24)
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            self.maximize_button.configure(text="□")
            self._is_maximized = False
            self._restore_panel_geometry = None

    def begin_drag(self, event):
        self._drag_origin = (event.x_root - self.root.winfo_x(),
                             event.y_root - self.root.winfo_y())

    def drag_window(self, event):
        if self._drag_origin is None:
            return
        x = event.x_root - self._drag_origin[0]
        y = event.y_root - self._drag_origin[1]
        self.root.geometry(f"+{x}+{y}")

    def startup_greeting_and_wake_word(self):
        """Speak the startup greeting first, then begin wake-word listening."""
        if self.closing:
            return

        greeting = (
            "Vanakkam! Naan JARVIS. Ready ah irukken. Thevai na Hey JARVIS nu koopidunga."
            if self.language.get() == "Tanglish"
            else "Hello! I'm JARVIS. Your personal assistant is ready. Say Hey JARVIS whenever you need me."
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

        # The dashboard may already run the monitor as its own process. Reuse
        # its fresh status heartbeat instead of starting a second writer.
        database_path = Path(__file__).resolve().parent / "data" / "routine.db"
        try:
            with sqlite3.connect(database_path, timeout=2) as connection:
                row = connection.execute(
                    "SELECT observed_at FROM monitor_status WHERE id = 1"
                ).fetchone()
            if row and (datetime.now() - datetime.fromisoformat(row[0])).total_seconds() <= 25:
                self.status.set("Using the daily monitor's screen tracker.")
                return
        except (sqlite3.Error, ValueError, OSError):
            pass

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
        is_user = speaker.lower() in {"you", "user"}
        speaker_tag = "speaker_you" if is_user else "speaker_jarvis"
        message_tag = "message_you" if is_user else "message_jarvis"
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{speaker}\n", speaker_tag)
        self.chat.insert(tk.END, f"{message}\n\n", message_tag)
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def clear_entry_placeholder(self, _event=None):
        if self.entry.get() == self.entry_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.configure(fg="#ffffff")

    def restore_entry_placeholder(self, _event=None):
        if not self.entry.get().strip():
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self.entry_placeholder)
            self.entry.configure(fg="#9badc2")

    def send_message(self):
        message = self.entry.get().strip()
        if not message or message == self.entry_placeholder:
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
        simple_text = re.sub(r"[^a-z0-9 ]", " ", message.lower())
        simple_text = re.sub(r"\s+", " ", simple_text).strip()
        asks_time = re.search(
            r"\b(?:what s (?:the )?time|what is (?:the )?time|what time is it|"
            r"tell me (?:the )?time|current time|time now|time enna|ippo time|"
            r"mani enna|time sollu)\b",
            simple_text,
        )
        if asks_time:
            current_time = datetime.now().strftime("%I:%M %p").lstrip("0")
            return (
                f"Ippo time {current_time}." if language == "Tanglish"
                else f"It is {current_time} right now."
            )

        asks_date = re.search(r"\b(?:what is the date|what s the date|today s date|what date is it)\b", simple_text)
        if asks_date:
            today = datetime.now().strftime("%A, %B %d")
            return f"Innaikku {today}." if language == "Tanglish" else f"Today is {today}."

        if re.search(r"\b(?:maximize|expand|full screen)\s+(?:the\s+)?jarvis\b", simple_text):
            self.root.after(0, self.toggle_maximize)
            return "JARVIS panel-a perusa open panren." if language == "Tanglish" else "Maximizing the JARVIS panel."

        if re.search(r"\b(?:quit|exit|shut down|shutdown|turn off)\s+(?:the\s+)?(?:jarvis|assistant)(?:\s+completely)?\b", simple_text):
            self._shutdown_after_speech = True
            return "Seri, JARVIS-a full-ah shut down panren." if language == "Tanglish" else "Okay, shutting JARVIS down after this reply."
        if re.search(r"\b(?:hide|close|minimize)\s+(?:the\s+)?(?:jarvis|assistant)\b", simple_text):
            self.root.after(0, self.hide_to_background)
            return "Seri, background-la irukken. Hey JARVIS nu koopidunga." if language == "Tanglish" else "I'll stay in the background. Say Hey JARVIS to open me again."

        gear_match = re.search(
            r"\bgear\s*(?:number\s*)?(1|2|3|4|5|one|two|three|four|five|first|second|third|fourth|fifth)\b",
            message.lower(),
        )
        if gear_match:
            gear_word = gear_match.group(1)
            gear = {"one":1, "first":1, "two":2, "second":2, "three":3,
                    "third":3, "four":4, "fourth":4, "five":5, "fifth":5}.get(gear_word)
            self.change_gear(int(gear_word) if gear_word.isdigit() else gear, skip_chat_speech=True)
            return f"Luffy changed to Gear {self.current_gear}!"

        if re.search(r"\b(say|speak|tell me)\b.*\b(thank you|thanks)\b", message.lower()):
            self.say_character_line("ありがとう！", "Thank you!", skip_chat_speech=True)
            return "ありがとう！"
        if re.search(r"\b(say|speak|tell me)\b.*\b(yo|hello|hi)\b", message.lower()):
            self.say_character_line("よー！", "Yo!", skip_chat_speech=True)
            return "よー！"

        # Understand natural window/app commands and launch apps from
        # Desktop/Start Menu shortcuts through the shared safe handler.
        shared_answer = handle_shared_desktop_command(message, language)
        if shared_answer is not None:
            return shared_answer
        text = message.strip().lower()
        # ASR may mishear "Jarvis" as "Javis"/"Jervis" and add filler words.
        # Normalize only a fixed set of safe commands; never execute arbitrary text.
        text = re.sub(
            r"^(?:(?:hey|hi|okay|ok)\s+)?(?:jarvis|javis|jervis|jarvas)[\s,.:;!?-]*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"\bplease\b", "", text).strip()
        text = re.sub(r"\s+(?:tool|app|application|program|please)$", "", text).strip()
        text = re.sub(r"^(?:can you|could you|would you)\s+", "", text).strip()
        tanglish = language == "Tanglish"

        # Safe, fixed Windows window-management actions.
        # These commands never execute user-supplied shell text.
        minimize_commands = {
            "minimize this window", "minimize window", "minimize this",
            "minimize jarvis", "minimize assistant",
        }
        maximize_commands = {
            "maximize this window", "maximize window", "maximize this",
            "maximize jarvis", "maximize assistant",
        }
        desktop_commands = {
            "show desktop", "go to desktop", "display desktop",
            "minimize all windows", "show me the desktop",
        }
        chrome_commands = {
            "switch to chrome", "open chrome", "switch chrome",
            "bring chrome", "bring up chrome", "focus chrome",
        }

        if text in minimize_commands | maximize_commands | desktop_commands:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                if text in desktop_commands:
                    # Win+D toggles the desktop; key-up events avoid a stuck key.
                    VK_LWIN, VK_D, KEYUP = 0x5B, 0x44, 0x0002
                    user32.keybd_event(VK_LWIN, 0, 0, 0)
                    user32.keybd_event(VK_D, 0, 0, 0)
                    user32.keybd_event(VK_D, 0, KEYUP, 0)
                    user32.keybd_event(VK_LWIN, 0, KEYUP, 0)
                    return "Desktop shown." if not tanglish else "Desktop kaamikkiren."

                hwnd = user32.GetForegroundWindow()
                if not hwnd:
                    return "I couldn't identify the active window."
                SW_MINIMIZE, SW_MAXIMIZE = 6, 3
                if text in minimize_commands:
                    user32.ShowWindow(hwnd, SW_MINIMIZE)
                    return "Minimized the active window." if not tanglish else "Active window minimize panniten."
                user32.ShowWindow(hwnd, SW_MAXIMIZE)
                return "Maximized the active window." if not tanglish else "Active window maximize panniten."
            except Exception as exc:
                print(f"[JARVIS window control] {type(exc).__name__}: {exc}")
                return "I couldn't control that window."

        if text in chrome_commands:
            try:
                # Use common Windows install locations instead of relying on PATH.
                program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
                candidates = [
                    os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
                    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
                ]
                chrome_path = next((path for path in candidates if path and os.path.isfile(path)), None)
                if chrome_path:
                    subprocess.Popen([chrome_path])
                    return "Opening Chrome." if not tanglish else "Chrome open panren."
                return "Chrome executable was not found in the usual installation locations."
            except Exception as exc:
                print(f"[JARVIS Chrome control] {type(exc).__name__}: {exc}")
                return "I couldn't launch Chrome. Please check its installation."

        # Windows master-volume controls. These are fixed, local actions only.
        volume_up = {
            "increase volume", "volume up", "turn volume up",
            "raise volume", "make volume louder", "louder",
        }
        volume_down = {
            "decrease volume", "volume down", "turn volume down",
            "lower volume", "make volume quieter", "quieter",
        }
        mute_commands = {"mute", "mute volume", "mute audio", "silence volume"}
        unmute_commands = {"unmute", "unmute volume", "turn sound on", "enable sound"}

        if text in volume_up | volume_down | mute_commands | unmute_commands:
            try:
                from ctypes import POINTER, cast
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                speakers = AudioUtilities.GetSpeakers()
                # pycaw versions differ: newer releases expose EndpointVolume
                # directly, while older releases require Activate + cast.
                endpoint = getattr(speakers, "EndpointVolume", None)
                if endpoint is None:
                    interface = speakers.Activate(
                        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                    )
                    endpoint = cast(interface, POINTER(IAudioEndpointVolume))

                if text in volume_up:
                    current = float(endpoint.GetMasterVolumeLevelScalar())
                    updated = min(1.0, current + 0.10)
                    endpoint.SetMasterVolumeLevelScalar(updated, None)
                    percent = round(updated * 100)
                    return (
                        f"Volume {percent} percent-ku increase panniten."
                        if tanglish else f"Volume increased to {percent} percent."
                    )

                if text in volume_down:
                    current = float(endpoint.GetMasterVolumeLevelScalar())
                    updated = max(0.0, current - 0.10)
                    endpoint.SetMasterVolumeLevelScalar(updated, None)
                    percent = round(updated * 100)
                    return (
                        f"Volume {percent} percent-ku decrease panniten."
                        if tanglish else f"Volume decreased to {percent} percent."
                    )

                if text in mute_commands:
                    endpoint.SetMute(1, None)
                    return "Audio mute panniten." if tanglish else "Audio muted."

                endpoint.SetMute(0, None)
                return "Audio unmute panniten." if tanglish else "Audio unmuted."

            except Exception as exc:
                print(f"[JARVIS volume control] {type(exc).__name__}: {exc}")
                return (
                    "Volume control work aagala. Windows audio device check pannunga."
                    if tanglish else
                    "I couldn't control the volume. Please check your Windows audio device."
                )

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
            "open paint tool": ("mspaint.exe", "Paint"),
            "start paint": ("mspaint.exe", "Paint"),
            "start paint tool": ("mspaint.exe", "Paint"),
            "paint open": ("mspaint.exe", "Paint"),
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

    def toggle_always_on_top(self):
        """Keep the floating assistant above other windows, or release it."""
        self.always_on_top = not self.always_on_top
        self.root.attributes("-topmost", self.always_on_top)
        self.pin_button.configure(text="Unpin" if self.always_on_top else "Pin")
        state = "pinned above other windows" if self.always_on_top else "no longer pinned"
        self.status.set(f"JARVIS is {state}.")

    def display_answer(self, answer):
        if self.closing:
            return

        self.last_answer = answer
        self.add_message("JARVIS", answer)

        if self._skip_next_speech:
            self._skip_next_speech = False
            return

        if self._shutdown_after_speech:
            self._shutdown_after_speech = False
            self.status.set("Shutting JARVIS down...")
            threading.Thread(
                target=self.speak_then_close,
                args=(answer, self.language.get()),
                name="JARVISShutdownSpeech",
                daemon=True,
            ).start()
            return

        # Automatically speak every answer. The wake listener pauses while speech plays,
        # then returns to waiting for the next "Hey JARVIS" phrase.
        self.status.set("Speaking...")
        threading.Thread(
            target=self.speak,
            args=(answer, self.language.get()),
            daemon=True,
        ).start()

    def speak_then_close(self, text, language):
        self.speak(text, language)
        if not self.closing:
            self.root.after(0, self.close)

    def hide_to_background(self):
        """Hide the window while leaving wake-word listening active."""
        if self.closing:
            return
        if not self.wake_enabled:
            self.toggle_wake_word()
        if not self.wake_enabled:
            return
        self._background_hidden = True
        self.root.withdraw()

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
                elif self.language.get() == "Japanese":
                    language_codes = ("ja-JP", "en-IN", "en-US")
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
        self.show_panel()
        self.root.lift()
        self.root.attributes("-topmost", self.always_on_top)
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
            "Help me decide what to do next using today's local routine records below. "
            "First state the task facts that were actually logged, then recommend at most "
            "three pending tasks in priority order. For each, give one concrete first step "
            "and a realistic time block. If no tasks were logged, say so plainly and ask "
            "me to choose one priority. Treat app usage only as foreground time, never as "
            "proof of work or task completion. Keep the answer short and conversational; "
            "do not use a table, HTML, or a long application-by-application dump.\n\n" + records
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
                else "ja-JP-NanamiNeural" if language == "Japanese"
                else "en-US-AnaNeural"
            )

            with self.speech_lock:
                if self.closing:
                    return

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
                    audio_path = audio_file.name

                async def generate_audio():
                    if language == "English":
                        communicator = edge_tts.Communicate(
                            speech_text, voice, rate="+8%", pitch="+12Hz"
                        )
                    else:
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
                    lang_code = "ta-IN" if language == "Tanglish" else "ja-JP" if language == "Japanese" else "en-IN"
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
