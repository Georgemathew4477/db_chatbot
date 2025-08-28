import os
import re
import time 
import math 
import tempfile
import numpy as np
from typing import List, Dict, Tuple
from groq import Groq
import platform
import shutil
import pytesseract


# Optional media processing
try:
    from faster_whisper import WhisperModel
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False

try:
    import ffmpeg
    HAVE_FFMPEG = True
except Exception:
    HAVE_FFMPEG = False

try:
    import easyocr
    HAVE_EASYOCR = True
except Exception:
    HAVE_EASYOCR = False

# Runtime toggles (env-driven)
USE_WHISPER = os.getenv("USE_WHISPER", "1") not in ("0", "false", "False")
USE_OCR = os.getenv("USE_OCR", "1") not in ("0", "false", "False")

# Optional explicit paths to ffmpeg/ffprobe binaries
FFMPEG_BIN = os.getenv("FFMPEG_PATH") or "ffmpeg"
FFPROBE_BIN = os.getenv("FFPROBE_PATH") or "ffprobe"

# --------------------------
# 0) LLM wrapper (Groq)
# --------------------------
def _chat(
    messages,
    api_key: str,
    model: str = "llama3-70b-8192",
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


# --------------------------
# 1) (Optional) Media -> Text
# --------------------------
_WHISPER = None
_OCR_READER = None


def _load_whisper():
    """Lazy-load faster-whisper model (CPU by default)."""
    global _WHISPER
    if _WHISPER is None and HAVE_WHISPER and USE_WHISPER:
        # Allow override via env, default to "small" for cloud
        model_size = os.getenv("WHISPER_MODEL", "small")  # "tiny"|"base"|"small"|"medium"|"large-v3"
        _WHISPER = WhisperModel(
            model_size,
            device="cpu",                # keep CPU on cloud
            compute_type="int8",         # best for CPU
            cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
            download_root=os.getenv("WHISPER_CACHE", "./.whisper")  # cache between runs
        )



def _load_ocr():
    """Lazy-load EasyOCR reader (CPU)."""
    global _OCR_READER
    if _OCR_READER is None and HAVE_EASYOCR and USE_OCR:
        _OCR_READER = easyocr.Reader(["en"], gpu=False)


def ocr_frames(video_path: str, step_sec: int = 3) -> str:
    """
    OCR across frames to capture on-screen text.
    Returns "" on any failure or when no video stream exists.
    """
    if not (HAVE_EASYOCR and HAVE_FFMPEG and USE_OCR):
        return ""
    _load_ocr()

    # Probe container
    try:
        probe = ffmpeg.probe(video_path, cmd=FFPROBE_BIN)
    except Exception:
        return ""

    # Find a video stream
    vstream = None
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            vstream = s
            break
    if vstream is None:
        return ""

    # Duration (from stream or format)
    duration = 0.0
    try:
        if vstream.get("duration"):
            duration = float(vstream["duration"])
        elif probe.get("format", {}).get("duration"):
            duration = float(probe["format"]["duration"])
    except Exception:
        duration = 0.0

    texts = []
    try:
        for t in np.arange(0, max(duration, 0), step_sec):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                (
                    ffmpeg.input(video_path, ss=float(t))
                    .filter("scale", 1280, -1)
                    .output(tmp.name, vframes=1, loglevel="error")
                    .overwrite_output()
                    .run(cmd=FFMPEG_BIN)
                )
                result = _OCR_READER.readtext(tmp.name, detail=0)
                os.unlink(tmp.name)
                if result:
                    texts.append(" ".join(result))
    except Exception:
        pass

    return _clean_text(" ".join(texts))


def _run_whisper_on_path(path: str) -> str:
    """Run faster-whisper and return joined transcript text."""
    try:
        segments, _ = _WHISPER.transcribe(
            path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(
            seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()
        )
        return _clean_text(text)
    except Exception:
        return ""

def _whisper_transcribe_path(path: str, log=print):
    segments, _ = _WHISPER.transcribe(
        path,
        beam_size=10,
        best_of=5,
        temperature=0.0,                # start deterministic
        vad_filter=False,               # important: we chunk ourselves
        condition_on_previous_text=False,
        language="en",
        task="transcribe",
        chunk_length=30,                # give model decent context
    )
    out, last_end, count = [], 0.0, 0
    for seg in segments:
        txt = getattr(seg, "text", "").strip()
        if txt:
            out.append(txt)
        if hasattr(seg, "end") and seg.end:
            try:
                last_end = float(seg.end)
            except Exception:
                pass
        count += 1
    text = " ".join(out).strip()
    log(f"      segments={count}, last_end≈{last_end:.1f}s, chars={len(text)}")
    return text, last_end, count



def _probe_duration_seconds(path: str) -> float:
    """Return media duration in seconds using ffprobe. 0.0 on failure."""
    try:
        info = ffmpeg.probe(path, cmd=FFPROBE_BIN)
        dur = info.get("format", {}).get("duration")
        if dur:
            return float(dur)
    except Exception:
        pass
    return 0.0



def transcribe_audio_or_video(
    video_or_audio_path: str,
    debug: bool = True,
    ui_log=None,
    chunk_minutes: int = 2,     # ≈2-minute chunks
    keep_intermediate: bool = False
) -> str:
    """
    Robust STT:
      1) Convert to 16k mono WAV (force first audio stream)
      2) Chunk with small overlap + tail pad
      3) Transcribe each chunk (no VAD), join text
      4) Micro-tail pass to catch final words
      5) Log coverage vs duration
    """
    def _log(msg: str):
        if debug:
            print(msg, flush=True)
        if ui_log:
            try:
                ui_log(msg)
            except Exception:
                pass

    if not (HAVE_WHISPER and USE_WHISPER):
        _log("⚠️ Whisper disabled or unavailable.")
        return ""
    if not HAVE_FFMPEG:
        _log("❌ ffmpeg not available; cannot convert/split.")
        return ""

    _load_whisper()
    _log(f"🎙️ Transcribing: {video_or_audio_path}")

    # 1) Convert to stable mono 16k WAV and FORCE the first audio stream
    full_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            full_wav = tmp.name
        (
            ffmpeg
            .input(video_or_audio_path)
            .output(
                full_wav,
                ac=1, ar=16000, format="wav",
                **{"map": "a:0"},          # <-- ensure correct audio stream
                loglevel="error"
            )
            .overwrite_output()
            .run(cmd=FFMPEG_BIN)
        )
    except Exception as e:
        _log(f"❌ Failed to convert to WAV: {e!r}")
        return ""

    dur = _probe_duration_seconds(full_wav)
    _log(f"🔧 Converted to WAV @16k mono. Duration ≈ {dur:.1f}s")

    # 2) Chunking with overlap + final tail pad
    CHUNK_SEC = max(60, int(chunk_minutes * 60))
    OVERLAP_SEC = 2.0          # helps continuity across chunks
    FINAL_TAIL_PAD = 1.0       # extend last chunk slightly

    total_chunks = 1 if dur and dur <= CHUNK_SEC else int(math.ceil(dur / CHUNK_SEC)) if dur else 1
    _log(f"⛓️ Will transcribe in {total_chunks} chunk(s) of ~{CHUNK_SEC}s (overlap={OVERLAP_SEC}s, tail_pad={FINAL_TAIL_PAD}s)")

    transcripts = []
    last_seen_end = 0.0

    try:
        if total_chunks == 1:
            _log("⏱️ Single-shot transcription (no VAD).")
            t0 = time.time()
            text, last_end, _ = _whisper_transcribe_path(full_wav, log=_log)
            _log(f"✅ Done in {time.time()-t0:.2f}s | coverage≈{last_end:.1f}/{dur:.1f}s")
            last_seen_end = last_end
            transcripts.append(text)

            # 3) Micro-tail pass if we appear to stop early
            if dur - last_end > 0.5:
                tail_start = max(0.0, dur - 8.0)  # last 8 seconds
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ttmp:
                    tail_wav = ttmp.name
                try:
                    (
                        ffmpeg
                        .input(full_wav, ss=tail_start, t=dur - tail_start + FINAL_TAIL_PAD)
                        .output(tail_wav, ac=1, ar=16000, format="wav", loglevel="error")
                        .overwrite_output()
                        .run(cmd=FFMPEG_BIN)
                    )
                    _log(f"  🔁 Micro-tail pass from {tail_start:.1f}s … {dur:.1f}s")
                    tail_text, tail_end, _ = _whisper_transcribe_path(tail_wav, log=_log)
                    if tail_text and (tail_text not in text):
                        transcripts.append(tail_text)
                        last_seen_end = dur
                finally:
                    if not keep_intermediate and os.path.exists(tail_wav):
                        try: os.unlink(tail_wav)
                        except Exception: pass

        else:
            # 2b) Multiple chunks with overlap + tail pad on last
            for i in range(total_chunks):
                start_nominal = i * CHUNK_SEC
                ss = max(0.0, start_nominal - (OVERLAP_SEC if i > 0 else 0.0))
                nominal_len = CHUNK_SEC if (i < total_chunks - 1) else max(0.0, dur - start_nominal)
                t_len = nominal_len + (OVERLAP_SEC if i > 0 else 0.0) + (FINAL_TAIL_PAD if i == total_chunks - 1 else 0.0)
                if t_len <= 0:
                    break

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ctmp:
                    chunk_wav = ctmp.name
                try:
                    (
                        ffmpeg
                        .input(full_wav, ss=ss, t=t_len)
                        .output(chunk_wav, ac=1, ar=16000, format="wav", loglevel="error")
                        .overwrite_output()
                        .run(cmd=FFMPEG_BIN)
                    )
                    _log(f"  ▶️ Chunk {i+1}/{total_chunks} @ {ss:.1f}s … {ss+t_len:.1f}s (nominal start {start_nominal:.1f}s)")
                    t0 = time.time()
                    text, last_end, _ = _whisper_transcribe_path(chunk_wav, log=_log)
                    _log(f"    ✅ {len(text)} chars in {time.time()-t0:.2f}s | chunk_coverage≈{last_end:.1f}/{t_len:.1f}s")
                    last_seen_end = max(last_seen_end, ss + last_end)
                    transcripts.append(text)
                except Exception as ce:
                    _log(f"    ❌ Chunk {i+1} failed: {ce!r}")
                finally:
                    if not keep_intermediate and os.path.exists(chunk_wav):
                        try: os.unlink(chunk_wav)
                        except Exception: pass

            # 3) Micro-tail pass (belt & braces)
            if dur - last_seen_end > 0.5:
                tail_start = max(0.0, dur - 8.0)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ttmp:
                    tail_wav = ttmp.name
                try:
                    (
                        ffmpeg
                        .input(full_wav, ss=tail_start, t=dur - tail_start + FINAL_TAIL_PAD)
                        .output(tail_wav, ac=1, ar=16000, format="wav", loglevel="error")
                        .overwrite_output()
                        .run(cmd=FFMPEG_BIN)
                    )
                    _log(f"  🔁 Micro-tail pass from {tail_start:.1f}s … {dur:.1f}s")
                    tail_text, tail_end, _ = _whisper_transcribe_path(tail_wav, log=_log)
                    if tail_text:
                        transcripts.append(tail_text)
                        last_seen_end = dur
                finally:
                    if not keep_intermediate and os.path.exists(tail_wav):
                        try: os.unlink(tail_wav)
                        except Exception: pass

    finally:
        if not keep_intermediate and full_wav and os.path.exists(full_wav):
            try: os.unlink(full_wav)
            except Exception: pass

    # 4) Join + coverage log (with clamped 100% when within margin)
    final_text = " ".join([t for t in transcripts if t]).strip()
    _log(f"🧩 Combined transcript length: {len(final_text)} chars")
    if dur:
        margin = 0.5
        coverage = (last_seen_end / dur * 100.0) if dur else 0.0
        if abs(dur - last_seen_end) < margin and coverage < 100.0:
            coverage = 100.0
            last_seen_end = dur
        _log(f"📏 Overall coverage: reached ≈ {last_seen_end:.1f}s of {dur:.1f}s ({coverage:.1f}%)")
    return final_text





# --------------------------
# 2) Claim Extraction
# --------------------------
def extract_claims(raw_text: str, api_key: str) -> List[str]:
    """Turn a transcript/caption into numbered, atomic medical claims."""
    sys = (
        "Extract strictly checkable, atomic medical claims about diabetes/health from the text. "
        "Output only a numbered list (3-12 lines). Each claim should be short and declarative."
    )
    out = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": raw_text}],
        api_key=api_key,
        max_tokens=700,
        temperature=0.1,
    )
    claims = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        if s[0].isdigit():
            s = re.sub(r"^\d+[\.\)\]]\s*", "", s).strip()
            if s:
                claims.append(s)
    return claims[:12]


# --------------------------
# 3) Retrieval helpers
# --------------------------
def rewrite_for_retrieval(text: str, api_key: str) -> str:
    """Rewrite the claim using clinical terms found in guidelines."""
    sys = "Rewrite the claim using clinical terms found in guidelines. Keep the meaning. Output one line."
    out = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": text}],
        api_key=api_key,
        max_tokens=120,
        temperature=0.0,
    )
    return _clean_text(out)


def retrieve_evidence_for_claim(
    claim: str,
    embedder,
    index,
    chunks: List[str],
    k: int,
    api_key: str,
) -> List[Tuple[str, int]]:
    """
    Returns [(chunk_text, chunk_index), ...] for top-k results.
    """
    rewritten = rewrite_for_retrieval(claim, api_key=api_key)
    vec = embedder.encode([rewritten])[0]
    _, I = index.search(np.array([vec]), k=k)
    hits = []
    for i in I[0]:
        if 0 <= i < len(chunks):
            hits.append((chunks[i], i))
    return hits


# --------------------------
# 4) Adjudication
# --------------------------
def judge_claim(claim: str, evidence_snippets: List[str], api_key: str) -> Dict:
    """
    Decide: SUPPORTED / CONTRADICTED / INSUFFICIENT, based ONLY on provided evidence.
    """
    context = "\n\n---\n".join(evidence_snippets[:6]) if evidence_snippets else "(no evidence found)"
    sys = (
        "You are an NHS guideline checker. Decide if the claim is SUPPORTED, CONTRADICTED, "
        "or INSUFFICIENT based ONLY on the provided context. Provide a brief reason and cite phrases from context."
    )
    user = (
        f"Claim:\n{claim}\n\nContext:\n{context}\n\nReturn exactly:\n"
        "Verdict: <SUPPORTED|CONTRADICTED|INSUFFICIENT>\nReason: <1-3 lines>"
    )
    out = _chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        api_key=api_key,
        max_tokens=350,
        temperature=0.0,
    )

    verdict = "INSUFFICIENT"
    u = out.upper()
    if "SUPPORTED" in u:
        verdict = "SUPPORTED"
    elif "CONTRADICTED" in u:
        verdict = "CONTRADICTED"

    reason = out
    m = re.search(r"Reason\s*:\s*(.+)", out, flags=re.I | re.S)
    if m:
        reason = m.group(1).strip()

    return {"claim": claim, "verdict": verdict, "reason": reason}


# --------------------------
# 5) Aggregation
# --------------------------
def aggregate_verdict(per_claim: List[Dict]) -> Tuple[str, str]:
    total = max(1, len(per_claim))
    sup = sum(1 for r in per_claim if r["verdict"] == "SUPPORTED")
    con = sum(1 for r in per_claim if r["verdict"] == "CONTRADICTED")
    ins = sum(1 for r in per_claim if r["verdict"] == "INSUFFICIENT")

    if con > 0:
        return "🔴 RED", f"{con} contradicted, {sup} supported, {ins} insufficient."
    if sup >= int(0.6 * total) and con == 0:
        return "🟢 GREEN", f"{sup} supported, {ins} insufficient (no contradictions)."
    return "🟡 AMBER", f"{sup} supported, {ins} insufficient (no hard contradictions)."


# --------------------------
# 6) Orchestration
# --------------------------
def verify_text_against_nhs(
    text: str,
    embedder,
    index,
    chunks: List[str],
    sources: List[str],
    api_key: str,
    k: int = 5,
) -> Dict:
    raw = _clean_text(text)
    if not raw:
        return {
            "claims": [],
            "results": [],
            "badge": "🟡 AMBER",
            "summary": "Empty text received; unable to verify.",
            "transcript": ""
        }

    claims = extract_claims(raw, api_key=api_key)
    if not claims:
        return {
            "claims": [],
            "results": [],
            "badge": "🟡 AMBER",
            "summary": "No clear medical claims found in the text.",
            "transcript": raw
        }

    results = []
    for c in claims:
        ev_pairs = retrieve_evidence_for_claim(c, embedder, index, chunks, k=k, api_key=api_key)
        ev_texts = [p[0] for p in ev_pairs]
        judg = judge_claim(c, ev_texts, api_key=api_key)
        judg["evidence_idxs"] = [p[1] for p in ev_pairs]
        results.append(judg)

    badge, summary = aggregate_verdict(results)
    return {
        "claims": claims,
        "results": results,
        "badge": badge,
        "summary": summary,
        "transcript": raw
    }



def verify_media_file(
    file_path: str,
    embedder,
    index,
    chunks: List[str],
    sources: List[str],
    api_key: str,
    k: int = 5,
    use_ocr: bool = True,
) -> Dict:
    """
    Accepts path to video/audio file. Transcribes speech, OCRs frames (optional), then verifies.
    """
    transcript = ""
    overlay = ""

    # STT
    try:
        transcript = transcribe_audio_or_video(file_path) if (HAVE_WHISPER and USE_WHISPER) else ""
    except Exception:
        transcript = ""

    # OCR (only if video stream exists & libs available)
    try:
        overlay = ocr_frames(file_path) if (use_ocr and HAVE_EASYOCR and HAVE_FFMPEG and USE_OCR) else ""
    except Exception:
        overlay = ""

    combined = _clean_text(f"{transcript}\n\n{overlay}")
    if not combined:
        return {
            "claims": [],
            "results": [],
            "badge": "🟡 AMBER",
            "summary": "No speech or on-screen text detected; unable to verify. "
                       "Check ffmpeg/Whisper setup or provide a transcript.",
            "transcript": ""
        }

    out = verify_text_against_nhs(combined, embedder, index, chunks, sources, api_key=api_key, k=k)
    out["transcript"] = combined  # ensure transcript is present
    
    return out
