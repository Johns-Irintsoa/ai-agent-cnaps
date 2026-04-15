import json
import time
import logging
from dataclasses import asdict
from collections import Counter

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import CONFIG
from data_classificateur.DataModel import FileEntry, ClassificationResult
from data_classificateur.Extracting import download_file, extract_zip, extract_rar, extract_bytes
from data_classificateur.Scrapping import scrape_all_pages

log = logging.getLogger(__name__)


def build_llm() -> ChatOllama:
    return ChatOllama(
        model=CONFIG.ollama_model,
        base_url=CONFIG.ollama_base_url,
        temperature=0,
        format="json",
        num_predict=150,
        num_ctx=2048,
    )
 
def build_prompt(label: str, group: str, text: str) -> str:
    cats = "\n".join(f"- {c}" for c in CONFIG.categories)
    body = text.strip()[:CONFIG.max_text_chars] if text.strip() else "(texte non extractible)"
    return f"""Tu es un classificateur de documents administratifs CNaPS Madagascar.
 
Contexte :
- Groupe de la page : "{group}"
- Libellé du fichier : "{label}"
 
Catégories disponibles :
{cats}
 
Extrait du document :
\"\"\"
{body}
\"\"\"
 
Réponds UNIQUEMENT en JSON valide :
{{
  "categorie": "<une catégorie parmi la liste>",
  "confiance": <0.0 à 1.0>,
  "raison": "<explication courte en 1 phrase>"
}}"""
 
def classify(llm: ChatOllama, label: str, group: str, text: str) -> dict:
    raw = llm.invoke([HumanMessage(content=build_prompt(label, group, text))]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())
 
 
# ─────────────────────── Pipeline principal ──────────────────────
 
def process_entry(entry: FileEntry, llm: ChatOllama) -> list[ClassificationResult]:
    source_name = entry.file_url.split("/")[-1].split("?")[0]
 
    def err(msg: str, ftype: str = "") -> list[ClassificationResult]:
        return [ClassificationResult(
            page_url=entry.page_url, group_title=entry.group_title,
            file_label=entry.file_label, file_url=entry.file_url,
            file_type=ftype or entry.file_type, source_name=source_name,
            categorie="autre", confiance=0.0, raison="", extrait="", erreur=msg,
        )]
 
    def make_result(ftype, sname, cat, conf, raison, extrait) -> ClassificationResult:
        return ClassificationResult(
            page_url=entry.page_url, group_title=entry.group_title,
            file_label=entry.file_label, file_url=entry.file_url,
            file_type=ftype, source_name=sname,
            categorie=cat, confiance=conf, raison=raison, extrait=extrait,
        )
 
    try:
        raw_bytes, ftype = download_file(entry.file_url, entry.file_type)
    except httpx.HTTPStatusError as e:
        return err(f"HTTP {e.response.status_code}")
    except Exception as e:
        return err(str(e))
 
    # Archives
    if ftype in ("zip", "rar"):
        members = (extract_zip if ftype == "zip" else extract_rar)(raw_bytes)
        del raw_bytes
        if not members:
            return err(f"Archive {ftype.upper()} sans contenu extractible", ftype)
        results = []
        for mname, text in members:
            try:
                p = classify(llm, entry.file_label, entry.group_title, text)
                results.append(make_result(
                    ftype, f"{source_name} → {mname}",
                    p.get("categorie", "autre"), float(p.get("confiance", 0.0)),
                    p.get("raison", ""), text[:200],
                ))
                log.info(f"    [{mname}] → {p.get('categorie')} ({p.get('confiance',0):.0%})")
                time.sleep(CONFIG.request_delay)
            except Exception as e:
                results.append(make_result(ftype, f"{source_name} → {mname}",
                    "autre", 0.0, "", ""))
                results[-1].erreur = str(e)
        return results
 
    # XLS/XLSX : pas d'extraction texte possible, on classifie par label
    if ftype in ("xls", "xlsx"):
        del raw_bytes
        try:
            p = classify(llm, entry.file_label, entry.group_title, "")
            return [make_result(ftype, source_name,
                p.get("categorie", "autre"), float(p.get("confiance", 0.0)),
                p.get("raison", "") + " [via libellé]", "")]
        except Exception as e:
            return err(str(e), ftype)
 
    # PDF / DOCX / TXT
    text = extract_bytes(raw_bytes, ftype)
    del raw_bytes
    try:
        p = classify(llm, entry.file_label, entry.group_title, text)
        return [make_result(ftype, source_name,
            p.get("categorie", "autre"), float(p.get("confiance", 0.0)),
            p.get("raison", ""), text[:200])]
    except json.JSONDecodeError as e:
        return err(f"LLM JSON error: {e}", ftype)
    except Exception as e:
        return err(str(e), ftype)
 
 
def run() -> list[ClassificationResult]:
    entries = scrape_all_pages()
    if not entries:
        log.error("Aucun fichier trouvé.")
        return []
 
    llm = build_llm()
    all_results: list[ClassificationResult] = []
 
    for i, entry in enumerate(entries, 1):
        log.info(
            f"\n[{i}/{len(entries)}] [{entry.group_title}] "
            f"{entry.file_label[:55]}... ({entry.file_type})"
        )
        all_results.extend(process_entry(entry, llm))
        if i < len(entries):
            time.sleep(CONFIG.request_delay)
 
    return all_results
 
 
# ─────────────────────────── Sortie ──────────────────────────────
 
def save_json(results: list[ClassificationResult], path: str = "resultats_cnaps.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    log.info(f"JSON plat → {path}")
 
 
def save_grouped_json(results: list[ClassificationResult], path: str = "resultats_groupes.json"):
    """Format structuré page → groupe → fichiers (idéal pour ingestion RAG / Lucy)."""
    grouped: dict = {}
    for r in results:
        grp = grouped.setdefault(r.page_url, {}).setdefault(r.group_title, [])
        grp.append({
            "fichier": r.source_name, "label": r.file_label,
            "url": r.file_url, "type": r.file_type,
            "categorie": r.categorie, "confiance": r.confiance,
            "raison": r.raison, "erreur": r.erreur,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    log.info(f"JSON groupé → {path}")
 
 
def print_summary(results: list[ClassificationResult]):
    erreurs = [r for r in results if r.erreur]
    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ CNaPS — {len(results)} document(s) traité(s)")
    print(f"{'='*65}")
    print("\n  Par catégorie :")
    for cat, n in Counter(r.categorie for r in results).most_common():
        print(f"    {cat:<15} {'█'*n} ({n})")
    print("\n  Par type de fichier :")
    for ft, n in Counter(r.file_type for r in results).most_common():
        print(f"    {ft:<10} {n}")
    print("\n  Par page source :")
    for pg, n in Counter(r.page_url for r in results).most_common():
        print(f"    {pg.split('/')[-1]:<25} {n} fichier(s)")
    if erreurs:
        print(f"\n  ⚠  Erreurs : {len(erreurs)}")
        for r in erreurs[:5]:
            print(f"    - {r.source_name[:50]} : {r.erreur}")
        if len(erreurs) > 5:
            print(f"    ... et {len(erreurs)-5} autre(s)")
    print(f"{'='*65}\n")