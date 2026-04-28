import sys
import os

# On ajoute 'src' au PATH pour que tous les imports fonctionnent
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# On initialise le cache AVANT toute autre action
import config_env

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
