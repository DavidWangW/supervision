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

for directory in (UPLOAD_DIR, PREVIEW_DIR, OUTPUT_DIR, MODELS_DIR, DATA_DIR, SERVER_VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
