from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR
from app.db.database import init_db
from app.routers import (
    analytics,
    analyze,
    environment,
    health,
    records,
    speed,
    streams,
    videos,
)
from app.schemas.common import ApiInfo
from app.services.stream_manager import stream_manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield
    stream_manager.stop_all()


app = FastAPI(
    title="Supervision Video API",
    description="Upload videos for detection and tracking powered by supervision.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5175",
        "http://localhost:5175",
        "http://127.0.0.1:8005",
        "http://localhost:8005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(videos.router)
app.include_router(speed.router)
app.include_router(analyze.router)
app.include_router(records.router)
app.include_router(analytics.router)
app.include_router(streams.router)
app.include_router(environment.router)


@app.get("/api/info", response_model=ApiInfo)
def api_info() -> ApiInfo:
    """Return API metadata and documentation links."""
    return ApiInfo(
        message="Supervision Video API",
        docs="/docs",
        health="/health",
    )


WEBAPP_DIST = BASE_DIR / "webapp" / "dist"
if WEBAPP_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(WEBAPP_DIST), html=True),
        name="frontend",
    )
