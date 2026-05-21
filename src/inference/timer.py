"""
RAGTimer — mesure le temps de chaque étape du pipeline RAG.

Expose deux context managers :
  - measure()  : synchrone  (CPU pur, ex: RRF Fusion)
  - ameasure() : asynchrone (I/O réseau, ex: embedding, ChromaDB, LLM)
"""
import time
import logging
from contextlib import contextmanager, asynccontextmanager

logger = logging.getLogger(__name__)


class RAGTimer:
    def __init__(self):
        self.steps: dict = {}
        self.total_start = time.perf_counter()

    @contextmanager
    def measure(self, step_name: str):
        """Context manager synchrone — pour les étapes CPU (ex: RRF Fusion)."""
        start = time.perf_counter()
        yield
        self.steps[step_name] = time.perf_counter() - start

    @asynccontextmanager
    async def ameasure(self, step_name: str):
        """Context manager asynchrone — pour les étapes I/O (embed, Chroma, BM25, LLM)."""
        start = time.perf_counter()
        yield
        self.steps[step_name] = time.perf_counter() - start

    def report(self) -> dict:
        """Affiche le rapport de timing dans les logs et retourne les mesures sous forme de dict."""
        total = time.perf_counter() - self.total_start

        logger.info("\n" + "=" * 50)
        logger.info("RAG PIPELINE TIMING")
        logger.info("=" * 50)
        for step, duration in self.steps.items():
            pct = (duration / total) * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            logger.info(f"{step:25s} {duration:7.3f}s  {pct:5.1f}%  {bar}")
        logger.info("-" * 50)
        logger.info(f"{'TOTAL':25s} {total:7.3f}s  100.0%")
        logger.info("=" * 50)

        return {**self.steps, "total": round(total, 3)}
