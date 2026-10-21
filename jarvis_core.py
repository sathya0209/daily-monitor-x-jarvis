
import os
import sqlite3
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "routine.db"

load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()


def get_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")
    return Groq(api_key=api_key)


# Keep a short conversation history so JARVIS remembers recent turns.
# The desktop app is a single-user local assistant, so in-memory history is enough.
_CONVERSATION_HISTORY = {"English": [], "Tanglish": [], "Japanese": []}
_MAX_HISTORY_MESSAGES = 10


def reset_conversation(response_language=None):
    """Clear the recent conversation when the user returns JARVIS to standby."""
    if response_language in _CONVERSATION_HISTORY:
        _CONVERSATION_HISTORY[response_language].clear()
    else:
        for history in _CONVERSATION_HISTORY.values():
            history.clear()


def _clean_answer(text):
    """Remove common Markdown formatting that looks awkward in a voice assistant UI."""
    import re

    text = (text or "").strip()
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def _search_web_items(query, max_results=5):
    """Return normalized search results with source domains for cross-checking."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results, region="in-en")
        items = []
        for item in results or []:
            title = (item.get("title") or "").strip()
            body = (item.get("body") or item.get("description") or "").strip()
            url = (item.get("href") or item.get("url") or "").strip()
            if not title or not (body or url):
                continue
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = "unknown source"
            items.append({"title": title, "body": body, "url": url, "domain": domain})
        return items
    except Exception as exc:
        print(f"[JARVIS web search] {type(exc).__name__}: {exc}")
        return []


def _search_web(query, max_results=5):
    """Compatibility wrapper returning readable snippets."""
    items = _search_web_items(query, max_results=max_results)
    return "\n\n".join(
        f"Title: {item['title']}\nSnippet: {item['body']}\nURL: {item['url']}"
        for item in items
    )


def _normalize_spoken_query(message, history=None):
    """
    Normalize common speech-to-text errors for search intent without changing
    unrelated words. Keep the original user message for conversation display.
    """
    text = re.sub(r"\s+", " ", (message or "")).strip()
    lower = text.lower()

    # Common transcription variants of "box-office collection".
    replacements = (
        (r"\bbox\s+of\s+collection\b", "box office collection"),
        (r"\bbox\s+office\s+collections?\b", "box office collection"),
        (r"\bbox\s+collection\b", "box office collection"),
        (r"\bcollection\s+of\s+(?:the\s+)?movie\b", "box office collection of the movie"),
        (r"\bworld\s+wide\s+collection\b", "worldwide box office collection"),
        (r"\bworldwide\s+collection\b", "worldwide box office collection"),
        (r"\bhow\s+much\s+(?:it\s+)?collected\b", "box office collection"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)

    # If the current turn clearly asks for collections and names a movie, keep the
    # movie title. If it is a short follow-up, resolve the title from recent turns.
    collection_intent = any(term in text.lower() for term in (
        "box office", "collection", "collected", "gross", "grossed", "earnings"
    ))
    if collection_intent:
        movie = re.search(r"\b(jailer\s*2|jailer\s+two)\b", text, flags=re.I)
        if movie:
            text = re.sub(r"\bjailer\s*2\b|\bjailer\s+two\b", "Jailer 2", text, flags=re.I)
        elif history:
            recent = " ".join(str(x.get("content", "")) for x in history[-6:])
            if re.search(r"\bjailer\s*2\b|\bjailer\s+two\b", recent, flags=re.I):
                # Only add this context when the user has not supplied another title.
                has_other_title = bool(re.search(
                    r"\b(jana\s+nayagan|vettaiyan|coolie|jailer|2\.0|kabali)\b",
                    text, flags=re.I
                ))
                if not has_other_title:
                    text = f"{text} for the movie Jailer 2"

    # A speech transcript may render "Jailer 2" as "Jailer two".
    text = re.sub(r"\bjailer\s+two\b", "Jailer 2", text, flags=re.I)
    return text.strip()


def _needs_fresh_search(message, history=None):
    """Detect current claims and movie box-office follow-ups that need search."""
    text = _normalize_spoken_query(message, history).lower()
    freshness_terms = (
        "latest", "current", "right now", "today", "this week", "recent",
        "trending", "trend", "news", "newest", "release date", "box office",
        "box-office", "collection", "collected", "gross", "grossed", "earnings",
        "who won", "score", "price", "updated", "as of now", "what happened",
        "2026", "2025", "who sang", "singer", "song", "soundtrack", "music",
        "movie", "film", "identify", "what is the name", "which song",
        "which movie", "actor", "actress", "lyrics", "weather", "stock",
        "election", "cricket", "football", "technology", "ai model", "anirudh",
        "rajinikanth", "rajini", "release", "schedule", "available", "version",
        "just launched", "newly launched", "prototype", "flying car", "tesla",
        "announced", "announcement", "launch", "unveiled", "revealed",
        "is it true", "did they", "has it", "new product", "demo video",
    )
    if any(term in text for term in freshness_terms):
        return True

    followup_terms = (
        "specific", "that one", "this one", "it", "they", "that", "really",
        "are you sure", "check it", "verify", "look it up", "just launched",
        "i saw", "i heard", "the prototype", "the video", "how much", "how many",
    )
    if history and any(term in text for term in followup_terms):
        recent_context = " ".join(
            str(item.get("content", "")) for item in history[-6:]
        ).lower()
        current_context_terms = (
            "launch", "prototype", "flying car", "tesla", "latest", "recent",
            "news", "announced", "released", "current", "trending", "video",
            "jailer", "rajini", "rajinikanth", "movie", "box office", "collection",
        )
        if any(term in recent_context for term in current_context_terms):
            return True
    return False


def _make_search_query(message, history):
    """Turn a spoken question into a standalone query, correcting common ASR variants."""
    normalized_message = _normalize_spoken_query(message, history)
    # For an explicit box-office question, deterministic query construction avoids
    # an LLM reinterpreting "box of collection" as an unrelated place or app.
    if any(term in normalized_message.lower() for term in (
        "box office", "collection", "collected", "gross", "grossed", "earnings"
    )):
        movie = re.search(
            r"\b(jailer\s*2|jailer\s+two)\b",
            normalized_message,
            flags=re.I,
        )
        if movie:
            return f"Jailer 2 worldwide box office collection latest reported total October 2026"
        if not history:
            return normalized_message
    if not history:
        return normalized_message
    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the latest message into one concise, standalone web-search query. "
                        "Use history only to resolve references such as it, that, or who is. "
                        "If the latest message clearly names a new topic or person, search "
                        "that topic only; do not mix in unrelated earlier topics. "
                        "For recent movie questions, include the named actor and latest movie. "
                        "Do not answer. Return only the search query, no explanation or markdown."
                    ),
                },
                *history[-6:],
                {"role": "user", "content": normalized_message},
            ],
            temperature=0.2,
            max_completion_tokens=80,
        )
        query = _clean_answer(response.choices[0].message.content or "")
        query = re.sub(r"[\r\n]+", " ", query).strip(" \"'")
        if query and len(query) < 300:
            print(f"[JARVIS search query] {query}")
            return query
    except Exception as exc:
        print(f"[JARVIS query rewrite] {type(exc).__name__}: {exc}")
    # A plain fallback still includes the latest turns so short follow-ups are searchable.
    context = " ".join(
        str(item.get("content", "")) for item in history[-4:]
    )
    return (context + " " + normalized_message)[-700:]


def _ask_with_verified_web(message, response_language, history):
    """Search with several focused queries, deduplicate sources, then answer cautiously."""
    normalized_message = _normalize_spoken_query(message, history)
    search_query = _make_search_query(normalized_message, history)

    # Try focused alternatives. This avoids relying on one long conversational query.
    candidates = [
        search_query,
        normalized_message,
        f"{search_query} official announcement",
        f"{search_query} latest news credible sources",
    ]
    unique_candidates = []
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate and candidate.lower() not in {q.lower() for q in unique_candidates}:
            unique_candidates.append(candidate)

    gathered = []
    seen_urls = set()
    for candidate in unique_candidates[:4]:
        print(f"[JARVIS search query] {candidate}")
        items = _search_web_items(candidate, max_results=6)
        print(f"[JARVIS search results] {len(items)} result(s) for query")
        for item in items:
            identity = (item["url"] or (item["domain"] + item["title"])).lower()
            if identity in seen_urls:
                continue
            seen_urls.add(identity)
            item["query"] = candidate
            gathered.append(item)
        # Continue with another query to get independent sources, but stop once we
        # have enough distinct domains and a useful number of results.
        domains = {x["domain"] for x in gathered if x["domain"] and x["domain"] != "unknown source"}
        if len(gathered) >= 7 and len(domains) >= 4:
            break

    if not gathered:
        raise RuntimeError("All web search attempts returned no results. Check internet access and ddgs.")

    # Put likely primary sources before general publishers; never call a source official
    # just because it ranks first in the search response.
    primary_terms = (".gov", "gov.in", "tesla.com", "sunpictures.in", "sunpictures.com",
                     "youtube.com", "x.com", "twitter.com", "instagram.com")
    gathered.sort(key=lambda x: (0 if any(t in x["domain"] for t in primary_terms) else 1,
                                 x["domain"], x["title"].lower()))
    domains = sorted({x["domain"] for x in gathered if x["domain"] and x["domain"] != "unknown source"})
    print(f"[JARVIS source domains] {', '.join(domains) if domains else 'none identified'}")
    print(f"[JARVIS verification] {len(gathered)} unique result(s) from {len(domains)} distinct domain(s)")

    evidence = "\n\n".join(
        f"Source {i}: {item['domain']}\nTitle: {item['title']}\nSnippet: {item['body']}\nURL: {item['url']}"
        for i, item in enumerate(gathered[:18], start=1)
    )

    if response_language == "Tanglish":
        language = (
            "Reply in casual, natural Chennai Tanglish written in English letters. "
            "Code-switch Tamil and English the way Chennai speakers do in everyday conversation. "
            "Use simple local phrasing such as 'seri', 'sollunga', 'pannalaam', and 'enna' when they fit; "
            "keep the tone friendly and relaxed. Avoid formal literary Tamil, forced slang, and overusing 'da/di'."
        )
    elif response_language == "Japanese":
        language = "Reply in natural, friendly spoken Japanese."
    else:
        language = "Reply in natural spoken Indian English."

    system_prompt = (
        "You are JARVIS, a careful personal voice assistant. Answer only the latest user "
        "message, using conversation history only to resolve references. The evidence below "
        "contains search-result snippets, NOT verified full articles. Snippets may be stale, "
        "wrong, promotional, or misleading. Never treat a search-result title alone as proof. "
        "For a current release, launch, announcement, or status, look for a clear date and "
        "prefer an official source plus independent reporting. For box-office collections, "
        "prices, scores, rankings, and other exact numbers, state a figure only when at least "
        "two different credible domains independently support the same figure and time period. "
        "If they report different numbers, say that estimates differ and do not choose one "
        "without explaining the discrepancy. Multiple pages from one domain count as one source. "
        "Do not infer that a movie has released merely because a scheduled date has passed. "
        "Distinguish announced, scheduled, released, postponed, cancelled, rumour, trailer, "
        "prototype, and concept. If the snippets do not establish the answer, say you could not "
        "verify it from the available results; do not fill gaps using memory or plausible guesses. "
        "Do not invent dates, names, amounts, source checks, or citations, and do not claim to "
        "have opened an article. Give a direct answer in 1-4 short spoken sentences, without "
        "Markdown or bullets. " + language +
        "\n\nSEARCH QUERY/QUERIES AND RESULTS:\n" + evidence[:18000]
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": normalized_message})

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=350,
    )
    answer = _clean_answer(response.choices[0].message.content or "")
    if not answer:
        raise RuntimeError("Groq returned an empty answer.")
    return answer

def ask_jarvis(message, response_language="English"):
    if not message or not message.strip():
        return "I'm listening. What would you like to talk about?"

    response_language = response_language if response_language in _CONVERSATION_HISTORY else "English"
    history = _CONVERSATION_HISTORY[response_language]

    if response_language == "Tanglish":
        language_rule = (
            "Speak in casual, natural Chennai Tanglish, written in English letters. "
            "Code-switch Tamil and English like a Chennai local in everyday conversation. "
            "Use familiar phrases such as 'seri', 'sollunga', 'pannalaam', 'enna', or 'pa' only when they fit. "
            "Keep it warm and relaxed; avoid formal Tamil, forced slang, and overusing 'da/di'."
        )
    elif response_language == "Japanese":
        language_rule = "Speak in natural, friendly conversational Japanese. Keep answers concise and easy to say aloud."
    else:
        language_rule = (
            "Speak in natural, warm Indian English, like a helpful person having "
            "a live conversation. Use contractions and simple spoken wording. "
            "Keep the delivery lively and playful when it fits, while giving accurate, direct help. "
            "Do not force catchphrases or claim to be a fictional character."
        )

    needs_fresh_search = _needs_fresh_search(message, history)
    if needs_fresh_search:
        try:
            answer = _ask_with_verified_web(message, response_language, history)
            history.extend([
                {"role": "user", "content": message.strip()},
                {"role": "assistant", "content": answer},
            ])
            del history[:-_MAX_HISTORY_MESSAGES]
            return answer
        except Exception as exc:
            print(f"[JARVIS verified web search error] {type(exc).__name__}: {exc}")
            # Preserve context even when search fails, so the next follow-up can
            # still be interpreted correctly. Avoid storing duplicate turns.
            history.extend([
                {"role": "user", "content": message.strip()},
                {"role": "assistant", "content": "I could not verify this because web search failed."},
            ])
            if len(history) > _MAX_HISTORY_MESSAGES:
                del history[:-_MAX_HISTORY_MESSAGES]
            return (
                "My web search failed on this attempt, so I can't verify that yet. You can ask me to try the search again."
                if response_language == "English"
                else "ウェブ検索に失敗したため、まだ確認できません。もう一度検索するよう頼んでください。"
                if response_language == "Japanese"
                else "Indha attempt-la web search fail aayiduchu, adhanala ippo verify panna mudiyala. Search-a marubadi try panna sollunga."
            )

    system_prompt = (
        "You are JARVIS, the user's personal voice companion. This is an ongoing "
        "spoken conversation, not a formal AI report. Respond like a thoughtful, "
        "natural human conversation partner: direct, friendly, clear, and relaxed. "
        "Usually answer in 1 to 4 short spoken sentences, then ask a brief follow-up "
        "only when it genuinely fits. For a question that needs teaching or detail, "
        "explain it in small conversational chunks rather than dumping a long essay. "
        "Do not use Markdown, headings, bullet points, asterisks, emojis, or stage "
        "directions because the answer is displayed and spoken aloud. Never repeat "
        "the user's question unnecessarily. Remember the recent conversation and "
        "use that context for follow-up phrases like 'explain that more'. Do not "
        "pretend you can hear the user while you are speaking, and never claim an "
        "action was completed unless it actually was. If the user says thanks or "
        "makes a natural closing remark, reply briefly and warmly; do not force "
        "another question. Be honest when you do not know something. "
        + language_rule
    )

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": message.strip()},
            ],
            temperature=0.2,
            max_completion_tokens=350,
        )

        raw_answer = response.choices[0].message.content
        answer = _clean_answer(raw_answer)
        if not answer:
            return "Sorry, I missed that. Could you say it once more?"

        history.extend([
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": answer},
        ])
        if len(history) > _MAX_HISTORY_MESSAGES:
            del history[:-_MAX_HISTORY_MESSAGES]
        return answer

    except Exception as exc:
        # Keep the chat reply friendly, but log the actual cause for debugging.
        print(f"[JARVIS Groq error] {type(exc).__name__}: {exc}")
        return "Sorry, I couldn't get a response just now. Please try again in a moment."



def transcribe_audio(audio_path, language_mode="English"):
    """
    Transcribe speech using Groq Whisper.
    Tanglish mode allows language detection to help with mixed Tamil-English.
    """
    with open(audio_path, "rb") as audio_file:
        arguments = {
            "file": audio_file,
            "model": "whisper-large-v3-turbo",
            "response_format": "text",
            "temperature": 0.0,
        }

        if language_mode == "English":
            arguments["language"] = "en"
            arguments["prompt"] = (
                "The speaker uses Indian English. Preserve names and "
                "technical terms accurately."
            )
        else:
            arguments["prompt"] = (
                "The speaker uses Chennai Tamil mixed with English, commonly "
                "called Tanglish. The speech may contain Tamil words, English "
                "words, application names, and technical terms. Preserve the "
                "meaning and do not add words that were not spoken."
            )

        result = get_client().audio.transcriptions.create(**arguments)

    if isinstance(result, str):
        return result.strip()

    return getattr(result, "text", "").strip()


def prepare_speech_text(text, language_mode="English"):
    """Prepare Tanglish for the Tamil neural voice; English passes through unchanged."""
    if language_mode != "Tanglish" or not text.strip():
        return _clean_answer(text)

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the input as natural spoken Chennai Tamil in Tamil "
                        "script for a Tamil TTS voice. Preserve meaning exactly. "
                        "Keep common technical words and app names in English. "
                        "Use short, casual Chennai phrasing suitable for speech, without adding slang that changes meaning. "
                        "Return only the text."
                    ),
                },
                {"role": "user", "content": _clean_answer(text)},
            ],
            temperature=1.0,
            max_completion_tokens=250,
        )
        converted = response.choices[0].message.content
        return converted.strip() if converted else _clean_answer(text)
    except Exception:
        return _clean_answer(text)


def get_today_summary():
    """Return a factual, date-scoped report from the local routine database."""
    if not DB_PATH.exists():
        return "The local routine database does not exist yet."

    today = date.today().isoformat()
    lines = [f"DAILY MONITOR RECORDS — {today}",
             "Only records saved in the local database are included."]

    def format_duration(seconds):
        seconds = int(seconds or 0)
        minutes, remainder = divmod(seconds, 60)
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {remainder}s"
        return f"{remainder}s"

    try:
        with sqlite3.connect(DB_PATH, timeout=10) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            tables = {
                row["name"] for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            # Screen activity: restrict all totals to today's local date.
            if "screen_activity" in tables:
                activity_columns = {
                    row["name"] for row in cursor.execute(
                        "PRAGMA table_info(screen_activity)"
                    ).fetchall()
                }
                expected = {"start_time", "end_time", "app_name", "state", "duration_seconds"}
                if expected.issubset(activity_columns):
                    totals = cursor.execute(
                        """
                        SELECT state, COUNT(*) AS records,
                               COALESCE(SUM(duration_seconds), 0) AS seconds
                        FROM screen_activity
                        WHERE substr(start_time, 1, 10) = ?
                        GROUP BY state
                        ORDER BY state
                        """,
                        (today,),
                    ).fetchall()
                    total_records = cursor.execute(
                        "SELECT COUNT(*) FROM screen_activity WHERE substr(start_time, 1, 10) = ?",
                        (today,),
                    ).fetchone()[0]
                    coverage = cursor.execute(
                        """
                        SELECT MIN(start_time) AS first_start, MAX(end_time) AS last_end
                        FROM screen_activity
                        WHERE substr(start_time, 1, 10) = ?
                        """,
                        (today,),
                    ).fetchone()

                    lines.append("\nSCREEN ACTIVITY")
                    lines.append(f"Saved activity segments: {total_records}")
                    if coverage and coverage["first_start"]:
                        lines.append(f"First recorded segment: {coverage['first_start']}")
                        lines.append(f"Last recorded segment end: {coverage['last_end']}")
                    if not totals:
                        lines.append("No screen-activity records have been saved for today.")
                    else:
                        state_seconds = {str(row["state"]).lower(): int(row["seconds"] or 0) for row in totals}
                        active_seconds = state_seconds.get("active", 0)
                        idle_seconds = state_seconds.get("idle", 0)
                        lines.append(f"Recorded active time: {format_duration(active_seconds)}")
                        lines.append(f"Recorded idle time: {format_duration(idle_seconds)}")
                        lines.append(f"Total recorded time: {format_duration(active_seconds + idle_seconds)}")
                        lines.append("These are recorded durations, not proof of productivity.")

                    category_map = {}
                    if "app_categories" in tables:
                        category_columns = {
                            row["name"] for row in cursor.execute(
                                "PRAGMA table_info(app_categories)"
                            ).fetchall()
                        }
                        if {"app_name", "category"}.issubset(category_columns):
                            category_map = {
                                row["app_name"]: row["category"]
                                for row in cursor.execute(
                                    "SELECT app_name, category FROM app_categories"
                                ).fetchall()
                            }

                    app_rows = cursor.execute(
                        """
                        SELECT app_name, COALESCE(SUM(duration_seconds), 0) AS seconds,
                               COUNT(*) AS records
                        FROM screen_activity
                        WHERE substr(start_time, 1, 10) = ?
                          AND LOWER(state) = 'active'
                          AND UPPER(app_name) <> 'IDLE'
                        GROUP BY app_name
                        ORDER BY seconds DESC, app_name COLLATE NOCASE
                        LIMIT 20
                        """,
                        (today,),
                    ).fetchall()
                    lines.append("\nAPPLICATIONS (active-state records only)")
                    if app_rows:
                        for row in app_rows:
                            app_name = row["app_name"]
                            category = category_map.get(app_name, "Uncategorised")
                            lines.append(
                                f"- {app_name}: {format_duration(row['seconds'])}; "
                                f"{row['records']} records; configured category: {category}"
                            )
                    else:
                        lines.append("No active application records have been saved for today.")
                else:
                    lines.append("\nSCREEN ACTIVITY: The table exists, but its columns do not match the expected monitor schema.")
            else:
                lines.append("\nSCREEN ACTIVITY: No screen_activity table was found.")

            # Tasks: this project schema uses task/category/minutes/completed/entry_date.
            lines.append("\nTASKS RECORDED TODAY")
            if "tasks" not in tables:
                lines.append("No tasks table was found.")
            else:
                task_columns = {
                    row["name"] for row in cursor.execute(
                        "PRAGMA table_info(tasks)"
                    ).fetchall()
                }
                if {"task", "category", "minutes", "completed", "entry_date"}.issubset(task_columns):
                    task_rows = cursor.execute(
                        """
                        SELECT task, category, minutes, completed
                        FROM tasks
                        WHERE entry_date = ?
                        ORDER BY id
                        """,
                        (today,),
                    ).fetchall()
                    if not task_rows:
                        lines.append("No task records have been saved for today.")
                    else:
                        completed_count = 0
                        completed_minutes = 0
                        incomplete_count = 0
                        incomplete_minutes = 0
                        for row in task_rows:
                            value = str(row["completed"]).strip().lower()
                            done = value in {"1", "true", "yes", "completed"}
                            minutes = int(row["minutes"] or 0)
                            if done:
                                completed_count += 1
                                completed_minutes += minutes
                                status = "completed"
                            else:
                                incomplete_count += 1
                                incomplete_minutes += minutes
                                status = "incomplete"
                            lines.append(
                                f"- {row['task']} | category: {row['category'] or 'Uncategorised'} | "
                                f"recorded minutes: {minutes} | status: {status}"
                            )
                        lines.append(
                            f"Task totals: {len(task_rows)} records; "
                            f"{completed_count} completed ({completed_minutes} recorded minutes); "
                            f"{incomplete_count} incomplete ({incomplete_minutes} recorded minutes)."
                        )
                else:
                    lines.append("The tasks table exists, but its columns do not match the expected task/category/minutes/completed/entry_date schema.")

            lines.append("\nInterpretation note: application time and configured categories do not prove that a task was productive or completed.")
            return "\n".join(lines)

    except sqlite3.Error as exc:
        return f"Unable to read local records: {exc}"
