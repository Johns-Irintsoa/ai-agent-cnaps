import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes.chat import router as chat_router
from .routes.ingestion import router as ingestion_router
from .routes.deletion import router as deletion_router
from ..inference.cache.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await semantic_cache.connect()
    logger.info("Cache sémantique démarré")

    from ..inference.bm25_retriever import bm25_index
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, bm25_index.build)
    logger.info("BM25 index construit au démarrage")

    yield

    await semantic_cache.close()


app = FastAPI(title="AI Agent CNAPS", lifespan=lifespan)

app.include_router(chat_router)
app.include_router(ingestion_router)
app.include_router(deletion_router)
