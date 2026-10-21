"""Shared, allowlisted Windows actions used by both JARVIS interfaces.

Only fixed commands are supported. This module never executes arbitrary shell text.
"""
import os
import re
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path


def _shortcut_roots():
    """Return standard per-user and all-users shortcut locations on Windows."""
    roots = []
    home = Path.home()
    roots.extend((home / "Desktop", home / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs"))
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        roots.extend((Path(program_data) / "Desktop", Path(program_data) / "Microsoft/Windows/Start Menu/Programs"))
    return [root for root in roots if root.is_dir()]


def _open_installed_app(name):
    """Launch a uniquely matched Desktop/Start Menu shortcut, without a shell."""
    wanted = re.sub(r"\s+", " ", name.strip().lower())
    if not wanted:
        return False
    shortcuts = []
    for root in _shortcut_roots():
        try:
            shortcuts.extend(p for p in root.rglob("*.lnk") if p.is_file())
        except OSError:
            continue
    exact = [p for p in shortcuts if re.sub(r"\s+", " ", p.stem.lower()) == wanted]
    matches = exact or [p for p in shortcuts if wanted in re.sub(r"\s+", " ", p.stem.lower())]
    if len(matches) != 1:
        return False
    os.startfile(str(matches[0]))
    return True


def _natural_window_action(text):
    """Return (action, optional app name) for common spoken window requests."""
    cleaned = re.sub(r"^please\s+", "", text)
    cleaned = re.sub(r"^(?:can you|could you|would you|will you)\s+", "", cleaned)
    cleaned = re.sub(r"^(?:please\s+|jarvis\s+)+", "", cleaned)
    patterns = (
        ("maximize", r"(?:maximize|maximise|fullscreen|full screen|expand)(?:\s+(?:the\s+)?(.+?))?"),
        ("minimize", r"(?:minimize|minimise|hide|shrink)(?:\s+(?:the\s+)?(.+?))?"),
        ("restore", r"(?:restore|undo full.?screen)(?:\s+(?:the\s+)?(.+?))?"),
    )
    custom = re.match(r"^make\s+(.+?)\s+(full.?screen|bigger|larger)$", cleaned)
    if custom:
        target = re.sub(r"^(?:the\s+)?", "", custom.group(1)).strip()
        return "maximize", "" if target.lower() in {"this", "current", "active", "my window"} else target
    custom = re.match(r"^make\s+(.+?)\s+(smaller|small)$", cleaned)
    if custom:
        target = re.sub(r"^(?:the\s+)?", "", custom.group(1)).strip()
        return "minimize", "" if target.lower() in {"this", "current", "active", "my window"} else target
    custom = re.match(r"^make\s+(.+?)\s+normal$", cleaned)
    if custom:
        target = re.sub(r"^(?:the\s+)?", "", custom.group(1)).strip()
        return "restore", "" if target.lower() in {"this", "current", "active", "my window"} else target
    for action, verb in patterns:
        match = re.match(rf"^{verb}(?:\s+window)?[.!?]*$", cleaned)
        if match:
            target = (match.group(1) or "").strip()
            target = re.sub(r"\s+window$", "", target)
            if target.lower() in {"this", "current", "active", "my"}:
                target = ""
            target = re.sub(r"^(?:the\s+)?(?:current|active|this|my)\s+(?:app|application|window)$", "", target)
            target = re.sub(r"^(?:the\s+)?(?:app|application|window)$", "", target)
            return action, target.strip()
    return None


def _control_window(action, target=""):
    """Control the foreground window or a visible top-level window by title."""
    import ctypes
    user32 = ctypes.windll.user32
    hwnds = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def collect(hwnd, _):
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title, length + 1)
            hwnds.append((hwnd, title.value))
        return True

    user32.EnumWindows(callback_type(collect), 0)
    hwnd = user32.GetForegroundWindow()
    if target:
        needle = target.lower()
        candidates = [(handle, title) for handle, title in hwnds if needle in title.lower()]
        if not candidates:
            return False
        # Prefer an exact title match; otherwise use the first visible title match.
        hwnd = next((handle for handle, title in candidates if title.lower() == needle), candidates[0][0])
        user32.SetForegroundWindow(hwnd)
    if not hwnd:
        return False
    command = {"minimize": 6, "maximize": 3, "restore": 9}[action]
    user32.ShowWindow(hwnd, command)
    return True


def _volume_action(text):
    """Find a volume intent inside a conversational sentence, not only exact commands."""
    normalized = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    patterns = (
        ("up", r"\b(?:increase|raise|boost|turn up)\s+(?:(?:the|my|system|device|master)\s+)*volume\b"),
        ("up", r"\bvolume\s+(?:up|increase|higher|louder)\b"),
        ("up", r"\b(?:make|set)\s+(?:(?:the|my)\s+)?(?:volume|audio|sound)\s+(?:a bit\s+|a little\s+)?(?:louder|higher|up)\b"),
        ("up", r"\b(?:make it|turn it)\s+(?:a bit\s+|a little\s+)?(?:louder|up)\b"),
        ("down", r"\b(?:reduce|decrease|lower|turn down|quiet(?:en)?)\s+(?:(?:the|my|system|device|master)\s+)*volume\b"),
        ("down", r"\bvolume\s+(?:down|decrease|decrease|reduce|lower|quieter)\b"),
        ("down", r"\b(?:make|set)\s+(?:(?:the|my)\s+)?(?:volume|audio|sound)\s+(?:a bit\s+|a little\s+)?(?:quieter|softer|lower|down)\b"),
        ("down", r"\b(?:make it|turn it)\s+(?:a bit\s+|a little\s+)?(?:quieter|softer|lower|down)\b"),
    )
    matches = []
    for action, pattern in patterns:
        for match in re.finditer(pattern, normalized):
            preceding = normalized[max(0, match.start() - 32):match.start()]
            if re.search(r"\b(?:don't|dont|do not|never|shouldn't|shouldnt|should not|not)\s+(?:please\s+)?$", preceding):
                continue
            matches.append((match.end(), action))
    return max(matches)[1] if matches else None


def handle_shared_desktop_command(message, language="English"):
    """Run a known safe desktop command, or return None if it is not a command."""
    text = (message or "").strip().lower()
    text = re.sub(r"^(?:(?:hey|hi|okay|ok)\s+)?(?:jarvis|javis|jervis|jarvas)[\s,.:;!?-]*", "", text, flags=re.I)
    text = re.sub(r"\bplease\b", "", text).strip()
    text = re.sub(r"\s+(?:tool|app|application|program|please)$", "", text).strip()
    text = re.sub(r"^(?:can you|could you|would you)\s+", "", text).strip()
    tanglish = str(language).lower() == "tanglish"

    def say(en, ta):
        return ta if tanglish else en

    # Time/date are local and do not require the AI service.
    if any(p in text for p in ("what time", "current time", "tell me the time", "time now", "time enna", "mani enna")):
        return say(f"The time is {datetime.now().strftime('%I:%M %p')}.", f"Ippo time {datetime.now().strftime('%I:%M %p')}.")
    if any(p in text for p in ("what date", "today's date", "todays date", "what day is it", "date today", "innaiku enna date")):
        today = datetime.now().strftime("%A, %d %B %Y")
        return say(f"Today is {today}.", f"Innaiku {today}.")

    # Explicit web destinations.
    sites = {
        "open youtube": ("https://www.youtube.com", "YouTube"),
        "youtube open": ("https://www.youtube.com", "YouTube"),
        "open google": ("https://www.google.com", "Google"),
        "google open": ("https://www.google.com", "Google"),
        "open browser": ("https://www.google.com", "Google"),
        "open web browser": ("https://www.google.com", "Google"),
        "open gmail": ("https://mail.google.com", "Gmail"),
        "gmail open": ("https://mail.google.com", "Gmail"),
        "open chatgpt": ("https://chatgpt.com", "ChatGPT"),
        "chatgpt open": ("https://chatgpt.com", "ChatGPT"),
    }
    if text in sites:
        url, label = sites[text]
        webbrowser.open(url)
        return say(f"Opening {label}.", f"{label} open pannuren.")

    if text.startswith(("search web for ", "google search for ", "search google for ")):
        from urllib.parse import quote_plus
        query = re.sub(r"^(search web for |google search for |search google for )", "", text).strip()
        if query:
            webbrowser.open("https://www.google.com/search?q=" + quote_plus(query))
            return say(f"Searching Google for {query}.", f"Google-la {query} search pannuren.")

    # Fixed applications only.
    apps = {
        "open paint": ("mspaint.exe", "Paint"), "open paint tool": ("mspaint.exe", "Paint"),
        "start paint": ("mspaint.exe", "Paint"), "paint open": ("mspaint.exe", "Paint"),
        "open notepad": ("notepad.exe", "Notepad"), "start notepad": ("notepad.exe", "Notepad"),
        "open calculator": ("calc.exe", "Calculator"), "start calculator": ("calc.exe", "Calculator"),
        "open file explorer": ("explorer.exe", "File Explorer"),
        "start file explorer": ("explorer.exe", "File Explorer"),
        "open task manager": ("taskmgr.exe", "Task Manager"),
        "start task manager": ("taskmgr.exe", "Task Manager"),
    }
    if text in {"open chrome", "switch to chrome", "switch chrome", "bring chrome", "bring up chrome", "focus chrome"}:
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = [os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
                      os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
                      os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe")]
        chrome = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not chrome:
            return say("Chrome executable was not found in the usual installation locations.", "Chrome install location kandupidikka mudiyala.")
        subprocess.Popen([chrome])
        return say("Opening Chrome.", "Chrome open pannuren.")
    if text in apps:
        executable, label = apps[text]
        try:
            subprocess.Popen([executable])
            return say(f"Opening {label}.", f"{label} open pannuren.")
        except Exception:
            return say(f"I couldn't open {label}.", f"{label} open panna mudiyala.")

    if text in {"open settings", "open windows settings", "windows settings open"}:
        try:
            os.startfile("ms-settings:")
            return say("Opening Windows Settings.", "Windows Settings open pannuren.")
        except Exception:
            return say("I couldn't open Windows Settings.", "Windows Settings open panna mudiyala.")

    # Common folder shortcuts.
    home = os.path.expanduser("~")
    folders = {
        "open downloads": os.path.join(home, "Downloads"), "open my downloads": os.path.join(home, "Downloads"),
        "downloads open": os.path.join(home, "Downloads"), "open documents": os.path.join(home, "Documents"),
        "open my documents": os.path.join(home, "Documents"), "open desktop folder": os.path.join(home, "Desktop"),
        "open pictures": os.path.join(home, "Pictures"), "open my pictures": os.path.join(home, "Pictures"),
    }
    if text in folders:
        folder = folders[text]
        if os.path.isdir(folder):
            subprocess.Popen(["explorer.exe", folder])
            label = os.path.basename(folder) or "folder"
            return say(f"Opening {label}.", f"{label} open pannuren.")
        return say("That folder was not found.", "Andha folder kidaikkala.")

    # Launch apps installed with a shortcut in the Desktop or Start Menu.
    open_match = re.match(r"^(?:open|launch|start|run|bring up)\s+(?:the\s+)?(.+?)(?:\s+for me)?[.!?]*$", text)
    if open_match:
        app_name = re.sub(r"\s+(?:app|application|program)$", "", open_match.group(1)).strip()
        try:
            if _open_installed_app(app_name):
                return say(f"Opening {app_name}.", f"{app_name} open pannuren.")
        except OSError:
            return say(f"I couldn't open {app_name}.", f"{app_name} open panna mudiyala.")

    # Volume changes are constrained to +/-10%; recognize the intent anywhere
    # in a sentence so ordinary conversational wording works too.
    volume_action = _volume_action(text)
    mute = text in {"mute", "mute volume", "mute audio", "silence volume"}
    unmute = text in {"unmute", "unmute volume", "turn sound on", "enable sound"}
    if volume_action or mute or unmute:
        try:
            from ctypes import POINTER, cast
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            speakers = AudioUtilities.GetSpeakers()
            endpoint = getattr(speakers, "EndpointVolume", None)
            if endpoint is None:
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            if volume_action == "up":
                level = min(1.0, float(endpoint.GetMasterVolumeLevelScalar()) + 0.10)
                endpoint.SetMasterVolumeLevelScalar(level, None)
                return say(f"Volume increased to {round(level*100)} percent.", f"Volume {round(level*100)} percent-ku increase panniten.")
            if volume_action == "down":
                level = max(0.0, float(endpoint.GetMasterVolumeLevelScalar()) - 0.10)
                endpoint.SetMasterVolumeLevelScalar(level, None)
                return say(f"Volume decreased to {round(level*100)} percent.", f"Volume {round(level*100)} percent-ku decrease panniten.")
            if mute:
                endpoint.SetMute(1, None)
                return say("Audio muted.", "Audio mute panniten.")
            endpoint.SetMute(0, None)
            return say("Audio unmuted.", "Audio unmute panniten.")
        except Exception:
            return say("I couldn't control the volume. Please check your Windows audio device.", "Volume control work aagala. Windows audio device check pannunga.")

    # Fixed window actions; no arbitrary keyboard or shell input.
    window_action = _natural_window_action(text)
    show_desktop = {"show desktop", "go to desktop", "display desktop", "minimize all windows", "show me the desktop", "take me to the desktop"}
    if window_action or text in show_desktop:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            if text in show_desktop:
                VK_LWIN, VK_D, KEYUP = 0x5B, 0x44, 0x0002
                user32.keybd_event(VK_LWIN, 0, 0, 0); user32.keybd_event(VK_D, 0, 0, 0)
                user32.keybd_event(VK_D, 0, KEYUP, 0); user32.keybd_event(VK_LWIN, 0, KEYUP, 0)
                return say("Desktop shown.", "Desktop kaamikkiren.")
            action, target = window_action
            if not _control_window(action, target):
                return say("I couldn't find a visible window matching that app.", "Andha app window kandupidikka mudiyala.")
            description = {"minimize": "Minimized", "maximize": "Maximized", "restore": "Restored"}[action]
            subject = f" {target}" if target else " the active window"
            return say(f"{description}{subject}.", f"{subject.strip()} { {'minimize': 'minimize', 'maximize': 'maximize', 'restore': 'restore'}[action] } panniten.")
        except Exception:
            return say("I couldn't control that window.", "Window control panna mudiyala.")
    return None
