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

for directory in (UPLOAD_DIR, PREVIEW_DIR, OUTPUT_DIR, MODELS_DIR, DATA_DIR, SERVER_VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
