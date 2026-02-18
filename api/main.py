"""FastAPI application for the job search API."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

from src.data_loader import JobDataset
from src.embeddings import EmbeddingClient
from src.search_engine import SearchEngine

from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading dataset...")
    dataset = JobDataset.load()
    app.state.engine = SearchEngine(dataset)
    app.state.client = EmbeddingClient()
    logger.info("Ready — %s jobs indexed.", f"{len(dataset):,}")
    yield


app = FastAPI(title="HiringCafe Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve built frontend in production (when frontend/dist exists)
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
