import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

UPLOAD_DIR = BASE_DIR / "uploads"
PREVIEW_DIR = BASE_DIR / "previews"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "supervision.db"
# Bundled sample videos the web UI can offer when a user has no clip of their own.
SERVER_VIDEOS_DIR = APP_DIR / "videos"

# Common video containers accepted on upload. Browsers cannot decode many of
# these (e.g. .avi, .mkv, .wmv) natively, so the service transcodes them to a
# browser-playable H.264 MP4 before previewing.
ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ts",
}

# Map common video extensions to MIME types so downloads/previews report the
# correct content type even when the client sends a generic one.
EXTENSION_MIME_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".ts": "video/mp2t",
}

DEFAULT_WEIGHTS = MODELS_DIR / "yolo26x.pt"
DEFAULT_SPEED_WEIGHTS = MODELS_DIR / "yolo26x.pt"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.7

# Number of frames sent to the model in a single batched inference call when a
# GPU is available. Larger batches improve GPU utilization/throughput at the
# cost of more VRAM. On CPU, batching provides little benefit and is forced to
# 1 (see ``app.services.hardware.resolve_batch_size``).
#
# Configure via the ``SV_GPU_BATCH_SIZE`` environment variable:
#   * a positive integer (e.g. "32") uses that fixed batch size,
#   * "auto" (or "0") lets the service probe a few batches at runtime and pick
#     the throughput-optimal size for the current GPU/VRAM.
# A value of 0 is also the in-code sentinel for "auto-detect" (see
# ``app.services.hardware.resolve_effective_batch``).
# _SV_BATCH_ENV = os.environ.get("SV_GPU_BATCH_SIZE", "auto").strip().lower()
_SV_BATCH_ENV = os.environ.get("SV_GPU_BATCH_SIZE", "64").strip().lower()
if _SV_BATCH_ENV in ("auto", "0", "autodetect", ""):
    # 0 means "auto-detect optimal batch at runtime".
    DEFAULT_GPU_BATCH_SIZE = 0
else:
    DEFAULT_GPU_BATCH_SIZE = max(1, int(_SV_BATCH_ENV))

# --- Environment recognition backend (Phase 4) ------------------------------
# Which recognizer classifies weather / road condition / visibility:
#   * "heuristic" - classical-CV SceneClassifier only (offline, per-frame).
#   * "vlm"       - vision LLM labels preferred; heuristic fills in until the
#                   first VLM reading arrives and on VLM failures.
#   * "hybrid"    - heuristic runs per-frame for responsiveness while the VLM
#                   is sampled in the background and overrides the labels.
# "vlm" and "hybrid" currently share the same pipeline wiring (background
# sampling + label override); the distinction is kept for future divergence.
ENV_BACKEND = os.environ.get("SV_ENV_BACKEND", "hybrid").strip().lower()

# OpenAI-compatible endpoint serving the vision model (e.g. LMStudio).
VLM_BASE_URL = os.environ.get("SV_VLM_BASE_URL", "http://2.0.0.1:1234/v1").strip()
# Local LMStudio credential, baked in for convenience so the service runs out
# of the box. Override with the SV_VLM_API_KEY env var for other deployments;
# an empty value means no Authorization header is sent.
VLM_API_KEY = os.environ.get(
    "SV_VLM_API_KEY", "sk-lm-nQyPRt6h:3vXnES82Zx7D7tGpU4LC"
).strip()
# gemma-4-e4b-it (or gemma-4-26b-a4b-it or gemma-4-12b-it-qat) — local LMStudio vision model.
VLM_MODEL = os.environ.get("SV_VLM_MODEL", "gemma-4-12b-it-qat").strip()
# Per-request timeout (seconds). Warm calls answer in ~9-11 s locally, but a
# cold-loaded model can take a minute or more, so leave generous headroom.
VLM_TIMEOUT = float(os.environ.get("SV_VLM_TIMEOUT", "300"))
# Minimum spacing between background VLM samples in the video pipelines.
VLM_INTERVAL_SEC = float(os.environ.get("SV_VLM_INTERVAL_SEC", "10"))
# Editable instruction prompt for the VLM scene classifier. Operators can tune
# the wording / label guidance here without touching code; the classifier reads
# this file on every request (cached by mtime) and injects the label vocabularies
# from the constants above. Override with ``SV_VLM_PROMPT_FILE``.
VLM_PROMPT_FILE = os.environ.get(
    "SV_VLM_PROMPT_FILE", str(APP_DIR / "prompts" / "vlm_environment.txt")
)

# --- VLM request gating (viewer-aware) ----------------------------------
# When True, the (expensive) VLM environment recognition is only issued while
# at least one client is actively watching the stream (WebSocket / MJPEG). An
# unattended stream therefore does not keep hammering the local vision-LLM.
# Offline file analysis is unaffected (it always runs to completion). Set
# SV_VLM_ONLY_WHEN_VIEWED=false to restore the old always-on behavior.
VLM_ONLY_WHEN_VIEWED = (
    os.environ.get("SV_VLM_ONLY_WHEN_VIEWED", "true").strip().lower()
    in ("1", "true", "yes", "on")
)
# After the last viewer leaves, keep issuing VLM samples for this many seconds
# so a quick page / stream switch does not thrash the model with stop-start cycles.
VLM_VIEWER_GRACE_SEC = float(os.environ.get("SV_VLM_VIEWER_GRACE_SEC", "30"))
# A client is "actively watching" only if it has received live traffic
# (WebSocket snapshot / MJPEG frame) within the last VLM_VIEWER_HEARTBEAT_SEC
# seconds. Gating on this *heartbeat* (instead of a viewer reference count) is
# self-healing: a leaked/stuck viewer connection that no longer pulls frames
# cannot keep the local vision-LLM spinning, so an unattended stream idles
# within one interval instead of requesting VLM forever. Keep it comfortably
# above the WS push period (0.4 s) and the MJPEG frame cadence.
VLM_VIEWER_HEARTBEAT_SEC = float(os.environ.get("SV_VLM_VIEWER_HEARTBEAT_SEC", "5"))

for directory in (
    UPLOAD_DIR, PREVIEW_DIR, OUTPUT_DIR, MODELS_DIR, DATA_DIR, SERVER_VIDEOS_DIR
):
    directory.mkdir(parents=True, exist_ok=True)

# Ensure the editable prompt directory exists so the bundled default prompt can
# be written/served even on a fresh checkout.
(APP_DIR / "prompts").mkdir(parents=True, exist_ok=True)
