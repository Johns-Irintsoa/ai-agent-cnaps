import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

if os.environ.get("RUN_SCRAPPER"):
    from ingestion.scrap.scrapper import run_scrapper
    run_scrapper()

elif os.environ.get("RUN_CLASSIFICATION"):
    from data_classificateur.ClassificationLLM import (
        run,
        print_summary,
        save_json,
        save_grouped_json,
    )
    OUTPUT_DIR = "data/classification/list"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = run()
    if results:
        print_summary(results)
        save_json(results, f"{OUTPUT_DIR}/resultats_cnaps.json")
        save_grouped_json(results, f"{OUTPUT_DIR}/resultats_groupes.json")

else:
    import uvicorn
    if __name__ == "__main__":
        uvicorn.run(
            "api:app",
            host=os.environ["API_HOST"],
            port=int(os.environ["API_PORT"]),
            reload=False,
        )
