"""FastAPI application for the job search API."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.data_loader import JobDataset
from src.embeddings import EmbeddingClient
from src.search_engine import SearchEngine

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading dataset...")
    dataset = JobDataset.load()
    app.state.engine = SearchEngine(dataset)
    app.state.client = EmbeddingClient()
    print(f"Ready — {len(dataset):,} jobs indexed.")
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
