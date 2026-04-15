"""
Test de la fonction extract_rar corrigée (fichier temporaire).
Usage dans Docker :
    docker compose run --rm app python src/test_rar.py
"""
import sys
import os
import httpx

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from data_classificateur.Extracting import extract_rar

TEST_URL = (
    "https://www.cnaps.mg/fr/document/statistiques/"
    "statistique_2012-CNaPS_6011128fd0b177.98228863.rar"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CNaPS-Bot/1.0)"}


def main():
    print(f"Téléchargement de {TEST_URL} ...")
    resp = httpx.get(TEST_URL, headers=HEADERS, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    raw = resp.content
    print(f"Téléchargé : {len(raw)} octets\n")

    print("Extraction via extract_rar() ...")
    members = extract_rar(raw)

    if not members:
        print("ECHEC : aucun membre extrait.")
        return

    print(f"\nOK — {len(members)} fichier(s) extrait(s) :\n")
    for name, text in members:
        preview = text[:120].replace("\n", " ").strip()
        print(f"  [{name}]")
        print(f"    {len(text)} caractères extraits")
        print(f"    Aperçu : {preview!r}\n")


if __name__ == "__main__":
    main()
