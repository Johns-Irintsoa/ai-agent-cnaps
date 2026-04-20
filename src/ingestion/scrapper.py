"""
Scrapper de recuperation des donnees non structurees CNaPS.

Algorithme :
1. Lire cnaps_urls.json (18 URLs avec leurs classes CSS)
2. Pour chaque URL : scraper les liens de documents via scrape_all_pages_from_json()
3. Pour chaque lien :
   - Telecharger le fichier
   - Classifier la categorie via le LLM (data_classificateur)
   - Sauvegarder dans data/unstructured/{categorie}/
4. Pour les archives ZIP/RAR :
   - Extraire chaque membre document
   - Classifier chaque membre individuellement
   - Sauvegarder chaque membre dans data/unstructured/{categorie}/
"""

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ajoute src/ au path

import httpx

from config import CONFIG
from data_classificateur.ClassificationLLM import build_llm, classify
from data_classificateur.DataModel import FileEntry
from data_classificateur.Extracting import download_file, extract_bytes
from data_classificateur.Scrapping import scrape_all_pages_from_json
from data_classificateur.Utils import detect_type

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PROJECT_ROOT    = Path(__file__).resolve().parents[2]   # ai-agent-cnaps/
URLS_JSON_PATH   = _PROJECT_ROOT / "cnaps_urls.json"
UNSTRUCTURED_DIR = _PROJECT_ROOT / "data" / "unstructured"

IMAGE_TYPES          = {"png", "jpg", "jpeg", "gif"}
ARCHIVE_TYPES        = {"zip", "rar"}
DOC_TYPES            = {"pdf", "docx", "doc", "xls", "xlsx", "txt"} | IMAGE_TYPES
MIN_IMAGE_TEXT_CHARS = 50   # nb min de caracteres OCR pour considerer une image utile

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def filename_from_url(url: str) -> str:
    """Extrait le nom de fichier depuis une URL. Retourne 'fichier_inconnu' si vide."""
    name = os.path.basename(urllib.parse.urlparse(url).path)
    return name if name else "fichier_inconnu"


def get_output_path(categorie: str, ftype: str, filename: str) -> Path:
    """
    Retourne le chemin de destination data/unstructured/{categorie}/{ftype}/{filename}.
    Cree les repertoires si necessaire. Utilise 'autre' si la categorie est inconnue.
    """
    if categorie not in CONFIG.categories:
        categorie = "autre"
    dest_dir = UNSTRUCTURED_DIR / categorie / ftype
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / filename


def save_bytes(data: bytes, dest_path: Path) -> None:
    """Sauvegarde les bytes dans dest_path. Skip si le fichier existe deja (idempotence)."""
    if dest_path.exists():
        log.debug(f"  Deja present : {dest_path.name}, ignore")
        return
    dest_path.write_bytes(data)
    log.info(f"  Sauvegarde : {dest_path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Traitement des documents simples (PDF, DOCX, XLS, TXT, ...)
# ---------------------------------------------------------------------------

def process_document(entry: FileEntry, llm, seen_urls: set) -> bool:
    """
    Telecharge, classifie et sauvegarde un document non-archive.

    Args:
        entry:     FileEntry avec l'URL et les metadonnees du document.
        llm:       Instance ChatOllama de build_llm().
        seen_urls: Ensemble partage pour la deduplication cross-pages.

    Returns:
        True si sauvegarde avec succes, False en cas d'erreur ou de doublon.
    """
    if entry.file_url in seen_urls:
        log.debug(f"  Doublon ignore : {entry.file_url}")
        return False

    try:
        raw_bytes, ftype = download_file(entry.file_url, entry.file_type)
    except httpx.HTTPStatusError as e:
        log.warning(f"  HTTP {e.response.status_code} : {entry.file_url}")
        return False
    except Exception as e:
        log.warning(f"  Erreur telechargement {entry.file_url} : {e}")
        return False

    text = extract_bytes(raw_bytes, ftype)

    # Ignorer les images sans contenu textuel utile (simples photos, decorations)
    if ftype in IMAGE_TYPES and len(text.strip()) < MIN_IMAGE_TEXT_CHARS:
        log.info(f"  Image sans texte utile, ignoree : {filename_from_url(entry.file_url)}")
        return False

    try:
        result = classify(llm, entry.file_label, entry.group_title, text)
    except Exception as e:
        log.warning(f"  Erreur classification {entry.file_url} : {e}")
        return False

    categorie = result.get("categorie", "autre")
    confiance = result.get("confiance", 0.0)
    filename  = filename_from_url(entry.file_url)
    dest      = get_output_path(categorie, ftype, filename)

    save_bytes(raw_bytes, dest)
    seen_urls.add(entry.file_url)
    log.info(f"  [{ftype}] {filename} -> {categorie} ({confiance:.0%})")
    return True


# ---------------------------------------------------------------------------
# Traitement des archives ZIP
# ---------------------------------------------------------------------------

def process_zip_archive(raw_bytes: bytes, entry: FileEntry, llm) -> int:
    """
    Extrait, classifie et sauvegarde chaque membre document d'une archive ZIP.

    Args:
        raw_bytes: Bytes bruts de l'archive ZIP.
        entry:     FileEntry source (pour contexte LLM : group_title, file_label).
        llm:       Instance ChatOllama.

    Returns:
        Nombre de membres sauvegardes avec succes.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile:
        log.warning(f"  Archive ZIP invalide : {entry.file_url}")
        return 0

    saved = 0
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = os.path.splitext(info.filename.lower())[1]
            if ext not in {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"}:
                continue
            if info.file_size > CONFIG.archive_member_max:
                log.warning(f"  ZIP membre trop grand ({info.file_size // 1024} KB) : {info.filename}")
                continue

            try:
                member_bytes = zf.read(info.filename)
                ftype = ext.lstrip(".")
                if ftype == "doc":
                    ftype = "docx"
                text   = extract_bytes(member_bytes, ftype)
                bname  = os.path.basename(info.filename) or info.filename
                # Ignorer les images sans contenu textuel utile
                if ftype in IMAGE_TYPES and len(text.strip()) < MIN_IMAGE_TEXT_CHARS:
                    log.info(f"    ZIP [{bname}] image sans texte utile, ignoree")
                    continue
                label  = f"{entry.file_label} / {bname}"
                result = classify(llm, label, entry.group_title, text)
                categorie = result.get("categorie", "autre")
                confiance = result.get("confiance", 0.0)
                dest = get_output_path(categorie, ftype, bname)
                save_bytes(member_bytes, dest)
                saved += 1
                log.info(f"    ZIP [{bname}] -> {categorie} ({confiance:.0%})")
                time.sleep(CONFIG.request_delay)
            except Exception as e:
                log.warning(f"    ZIP membre {info.filename} erreur : {e}")

    return saved


# ---------------------------------------------------------------------------
# Traitement des archives RAR
# ---------------------------------------------------------------------------

def process_rar_archive(raw_bytes: bytes, entry: FileEntry, llm) -> int:
    """
    Extrait via unrar CLI, classifie et sauvegarde chaque membre document d'une archive RAR.

    Necessite le binaire `unrar` dans le PATH (defini dans le Dockerfile du projet).

    Args:
        raw_bytes: Bytes bruts de l'archive RAR.
        entry:     FileEntry source (pour contexte LLM).
        llm:       Instance ChatOllama.

    Returns:
        Nombre de membres sauvegardes avec succes.
    """
    tmp_rar = None
    tmp_dir = None
    saved   = 0

    try:
        # Ecrire l'archive dans un fichier temporaire (unrar necessite un chemin disque)
        with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_rar = tmp.name

        tmp_dir = tempfile.mkdtemp()
        result  = subprocess.run(
            ["unrar", "e", tmp_rar, tmp_dir + "/", "-y"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode > 1:
            log.warning(f"  RAR unrar erreur ({entry.file_url}) : {result.stderr.strip()}")
            return 0

        for fname in sorted(os.listdir(tmp_dir)):
            ext = os.path.splitext(fname.lower())[1]
            if ext not in {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"}:
                continue
            fpath = os.path.join(tmp_dir, fname)
            if os.path.getsize(fpath) > CONFIG.archive_member_max:
                log.warning(f"  RAR membre trop grand : {fname}")
                continue

            try:
                with open(fpath, "rb") as f:
                    member_bytes = f.read()
                ftype = ext.lstrip(".")
                if ftype == "doc":
                    ftype = "docx"
                text   = extract_bytes(member_bytes, ftype)
                # Ignorer les images sans contenu textuel utile
                if ftype in IMAGE_TYPES and len(text.strip()) < MIN_IMAGE_TEXT_CHARS:
                    log.info(f"    RAR [{fname}] image sans texte utile, ignoree")
                    continue
                label  = f"{entry.file_label} / {fname}"
                res    = classify(llm, label, entry.group_title, text)
                categorie = res.get("categorie", "autre")
                confiance = res.get("confiance", 0.0)
                dest = get_output_path(categorie, ftype, fname)
                save_bytes(member_bytes, dest)
                saved += 1
                log.info(f"    RAR [{fname}] -> {categorie} ({confiance:.0%})")
                time.sleep(CONFIG.request_delay)
            except Exception as e:
                log.warning(f"    RAR membre {fname} erreur : {e}")

    except Exception as e:
        log.warning(f"  RAR erreur globale ({entry.file_url}) : {e}")
    finally:
        if tmp_rar and os.path.exists(tmp_rar):
            os.unlink(tmp_rar)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return saved


# ---------------------------------------------------------------------------
# Dispatcher archives
# ---------------------------------------------------------------------------

def process_archive(entry: FileEntry, llm, seen_urls: set) -> int:
    """
    Telecharge une archive ZIP ou RAR et traite chaque membre document.

    Args:
        entry:     FileEntry de l'archive.
        llm:       Instance ChatOllama.
        seen_urls: Ensemble partage pour la deduplication.

    Returns:
        Nombre de membres sauvegardes (0 en cas d'erreur).
    """
    if entry.file_url in seen_urls:
        log.debug(f"  Archive doublon ignore : {entry.file_url}")
        return 0

    try:
        raw_bytes, ftype = download_file(entry.file_url, entry.file_type)
    except httpx.HTTPStatusError as e:
        log.warning(f"  HTTP {e.response.status_code} pour archive : {entry.file_url}")
        return 0
    except Exception as e:
        log.warning(f"  Erreur telechargement archive {entry.file_url} : {e}")
        return 0

    if ftype == "zip":
        count = process_zip_archive(raw_bytes, entry, llm)
    elif ftype == "rar":
        count = process_rar_archive(raw_bytes, entry, llm)
    else:
        log.warning(f"  Type archive inattendu '{ftype}' : {entry.file_url}")
        return 0

    if count > 0:
        seen_urls.add(entry.file_url)
    return count


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------

def run_scrapper() -> None:
    """
    Pipeline principal de recuperation des donnees non structurees CNaPS.

    Flux :
        cnaps_urls.json (18 URLs)
          -> scrape_all_pages_from_json()   [Scrapping.py etendu]
          -> pour chaque FileEntry :
               si archive  -> process_archive()  -> ZIP/RAR handler
               sinon       -> process_document() -> download + classify + save
          -> data/unstructured/{categorie}/{fichier}
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(message)s",
        )

    # Charger les entrees depuis cnaps_urls.json
    log.info(f"Chargement URLs depuis : {URLS_JSON_PATH}")
    entries = scrape_all_pages_from_json(str(URLS_JSON_PATH))
    log.info(f"Total entrees a traiter : {len(entries)}")

    if not entries:
        log.warning("Aucun document trouve. Verifier cnaps_urls.json et la connectivite reseau.")
        return

    llm       = build_llm()
    seen_urls: set = set()

    docs_saved     = 0
    archive_saved  = 0
    errors         = 0

    for i, entry in enumerate(entries, 1):
        log.info(f"\n[{i}/{len(entries)}] {entry.file_label[:60]} ({entry.file_type})")
        log.info(f"  URL : {entry.file_url}")

        if entry.file_type in ARCHIVE_TYPES:
            n = process_archive(entry, llm, seen_urls)
            if n > 0:
                archive_saved += n
            else:
                errors += 1
        elif entry.file_type in DOC_TYPES:
            ok = process_document(entry, llm, seen_urls)
            if ok:
                docs_saved += 1
            else:
                errors += 1
        else:
            log.debug(f"  Type non reconnu ignore : {entry.file_type}")

        time.sleep(CONFIG.request_delay)

    # Rapport final
    separator = "=" * 65
    print(f"\n{separator}")
    print(f"  RESUME SCRAPPER CNAPS")
    print(f"  Entrees trouvees      : {len(entries)}")
    print(f"  Documents sauvegardes : {docs_saved}")
    print(f"  Membres d archives    : {archive_saved}")
    print(f"  Erreurs               : {errors}")
    print(separator)
    print()
    for cat in CONFIG.categories:
        cat_dir = UNSTRUCTURED_DIR / cat
        if cat_dir.exists():
            n = len(list(cat_dir.iterdir()))
            if n > 0:
                print(f"  {cat:<15} {n} fichier(s)")
    print()


if __name__ == "__main__":
    run_scrapper()
