# utils/link_ingest.py
import os
import re
import tempfile
from typing import Dict, Optional

# Feature toggles
USE_YTDLP = os.getenv("USE_YTDLP", "1") not in ("0", "false", "False")
IS_CLOUD  = os.getenv("STREAMLIT_CLOUD", "1") not in ("0", "false", "False")

# Optional deps
try:
    import yt_dlp
    HAVE_YT_DLP = True
except Exception:
    HAVE_YT_DLP = False

try:
    import ffmpeg  # noqa: F401  (kept for future use)
    HAVE_FFMPEG = True
except Exception:
    HAVE_FFMPEG = False

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    FFPROBE_BIN = getattr(imageio_ffmpeg, "get_ffprobe_exe", lambda: "ffprobe")()
    HAVE_FFMPEG = True
except Exception:
    FFMPEG_BIN = os.getenv("FFMPEG_PATH") or "ffmpeg"
    FFPROBE_BIN = os.getenv("FFPROBE_PATH") or "ffprobe"

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    HAVE_YT_TRANSCRIPT = True
except Exception:
    HAVE_YT_TRANSCRIPT = False

# yt-dlp opts
YDL_COMMON_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "ignoreerrors": False,
    "retries": 5,
    "fragment_retries": 5,
    "socket_timeout": 20,
    "concurrent_fragment_downloads": 1,
    "noplaylist": True,
    "playlist_items": "1",
    "overwrites": True,
    "ffmpeg_location": FFMPEG_BIN if HAVE_FFMPEG else None,
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.youtube.com/",
        "Origin": "https://www.youtube.com",
    },
    "geo_bypass": True,
    "nocheckcertificate": True,
    # "cookiefile": os.getenv("YTDLP_COOKIES") or None,
}

def _file_ok(path: Optional[str]) -> bool:
    try:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 16_384
    except Exception:
        return False

# ---------- URL helpers ----------
def is_youtube_url(url: str) -> bool:
    return bool(re.search(r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)", url or "", re.I))

def is_youtube_shorts(url: str) -> bool:
    return bool(re.search(r"youtube\.com/shorts/", url or "", re.I))

def _extract_yt_id(url: str) -> Optional[str]:
    if not url:
        return None
    for pat in [
        r"youtu\.be/([A-Za-z0-9_\-]{6,})",
        r"[?&]v=([A-Za-z0-9_\-]{6,})",
        r"youtube\.com/shorts/([A-Za-z0-9_\-]{6,})",
    ]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

# ---------- Captions ----------
def fetch_youtube_transcript(yt_id: str, languages=("en", "en-US", "en-GB")) -> Optional[str]:
    if not (HAVE_YT_TRANSCRIPT and yt_id):
        return None
    try:
        # human captions first
        for lang in languages:
            try:
                segs = YouTubeTranscriptApi.get_transcript(yt_id, languages=[lang])
                if segs:
                    return " ".join(s.get("text", "").strip() for s in segs if s.get("text")).strip()
            except (NoTranscriptFound, TranscriptsDisabled):
                pass
        # auto captions
        try:
            lst = YouTubeTranscriptApi.list_transcripts(yt_id)
            auto = lst.find_generated_transcript(languages=list(languages))
            segs = auto.fetch()
            if segs:
                return " ".join(s.get("text", "").strip() for s in segs if s.get("text")).strip()
        except Exception:
            pass
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception):
        return None
    return None

# ---------- Description only ----------
def fetch_youtube_description(url: str) -> Optional[str]:
    if not (HAVE_YT_DLP and url):
        return None
    ydl_opts = {
        **YDL_COMMON_OPTS,
        "skip_download": True,
        "extract_flat": True,
        "forcejson": True,
        "dump_single_json": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            return None
        if info.get("_type") == "playlist" and info.get("entries"):
            info = info["entries"][0]
        title = (info.get("title") or "").strip()
        desc = (info.get("description") or "").strip()
        out = f"{title}\n\n{desc}".strip()
        return out or None
    except Exception:
        return None

# ---------- Audio download (best effort) ----------
def download_audio_with_ytdlp(url: str):
    # Skip in cloud unless explicitly enabled
    if IS_CLOUD and not os.getenv("ALLOW_CLOUD_YTDLP"):
        return {"error": "yt-dlp disabled on cloud (set ALLOW_CLOUD_YTDLP=1 to enable)"}
    if not (USE_YTDLP and HAVE_YT_DLP and url):
        return {"error": "yt-dlp not available"}

    tmpdir = tempfile.mkdtemp(prefix="yt-audio-")
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

    ydl_opts = {
        **YDL_COMMON_OPTS,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [],
        "merge_output_format": None,
        "paths": {"home": tmpdir, "temp": tmpdir},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if not info:
            return {"error": "no info from yt-dlp"}

        if info.get("_type") == "playlist":
            info = (info.get("entries") or [None])[0] or info

        vid = info.get("id") or ""
        candidate = None
        for fn in os.listdir(tmpdir):
            if fn.startswith(vid + "."):
                candidate = os.path.join(tmpdir, fn)
                break

        if candidate and _file_ok(candidate):
            return {
                "path": candidate,
                "id": vid,
                "title": (info.get("title") or "").strip(),
                "duration": info.get("duration"),
            }
        return {"error": "downloaded file missing/too small"}
    except Exception as e:
        return {"error": f"yt-dlp audio error: {e!r}"}

# ---------- Orchestrator ----------
def ingest_from_url(url: str) -> Dict:
    """
    - If NOT Shorts and captions exist → include as 'text'.
    - Try to download bestaudio if allowed; otherwise skip on cloud.
    - Always try description for extra context.
    - Never fail just because audio download failed.
    """
    url = (url or "").strip()
    if not url:
        return {"kind": "invalid", "text": None, "audio_path": None, "video_path": None, "note": "Empty URL."}

    if not is_youtube_url(url):
        return {"kind": "unsupported", "text": None, "video_path": None, "audio_path": None, "note": "Only YouTube is supported in this demo."}

    vid = _extract_yt_id(url)
    shorts = is_youtube_shorts(url)

    # Captions (non-Shorts)
    tx = fetch_youtube_transcript(vid) if (vid and not shorts) else None
    # Description
    desc = fetch_youtube_description(url)
    # Audio (best effort)
    ainfo = download_audio_with_ytdlp(url)
    audio_path = (ainfo or {}).get("path")
    dl_err = (ainfo or {}).get("error")

    note_bits = []
    if tx:   note_bits.append("Got official/auto captions.")
    if desc: note_bits.append("Fetched description.")
    if audio_path:
        note_bits.append(f"Downloaded bestaudio. id={(ainfo or {}).get('id', vid)} dur≈{(ainfo or {}).get('duration')}s")
    elif dl_err:
        note_bits.append(f"Audio download skipped/failed: {dl_err}")

    kind = (
        "youtube_transcript" if tx else
        "youtube_audio" if audio_path else
        "youtube_description" if desc else
        "youtube_fallback"
    )

    safe_text = tx if not shorts else None

    return {
        "kind": kind,
        "text": safe_text,
        "video_path": None,
        "audio_path": audio_path,
        "note": " ".join(note_bits) or f"Ingested YouTube id={vid or '?'}",
    }
