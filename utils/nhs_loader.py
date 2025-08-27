import json
from typing import List, Tuple

def load_chunks_and_sources(jsonl_path: str = "data/chunks.jsonl") -> Tuple[List[str], List[str]]:
    """
    Reads a JSONL with fields: {"text": "...chunk...", "source": "path-or-id"} per line.
    Returns (chunks, sources). Empty lists if file missing.
    """
    chunks, sources = [], []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunks.append(obj.get("text", ""))
                    sources.append(obj.get("source", "unknown"))
                except Exception:
                    continue
    except FileNotFoundError:
        return [], []
    return chunks, sources
