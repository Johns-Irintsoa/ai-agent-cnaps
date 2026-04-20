from pypdf import PdfReader
from docx import Document as DocxDocument
from io import BytesIO
import os, zipfile, rarfile, httpx, logging, tempfile, subprocess, shutil
from config import CONFIG
from data_classificateur.Utils import detect_type

rarfile.UNRAR_TOOL = "unrar"

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CNaPS-Bot/1.0)"}


def extract_pdf(raw: bytes) -> str:
    try:
        r = PdfReader(BytesIO(raw))
        return " ".join(p.extract_text() or "" for p in r.pages[:3]).strip()
    except Exception as e:
        log.debug(f"PDF error: {e}"); return ""
 
def extract_docx(raw: bytes) -> str:
    try:
        d = DocxDocument(BytesIO(raw))
        return " ".join(p.text for p in d.paragraphs[:20] if p.text.strip()).strip()
    except Exception as e:
        log.debug(f"DOCX error: {e}"); return ""
 
def extract_txt(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: continue
    return ""
 
def extract_image(raw: bytes, ftype: str = "png") -> str:
    """
    Extrait le texte d'une image via OCR (unstructured + tesseract).
    Ecrit les bytes dans un fichier temporaire car partition_image() requiert un chemin disque.
    Retourne une chaine vide si l'OCR echoue ou si l'image ne contient pas de texte.
    """
    tmp_img = None
    try:
        suffix = f".{ftype}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_img = tmp.name
        from unstructured.partition.image import partition_image
        elements = partition_image(filename=tmp_img, languages=["fra", "eng"])
        text = " ".join(e.text for e in elements if hasattr(e, "text") and e.text).strip()
        log.debug(f"  OCR image : {len(text)} caracteres extraits")
        return text
    except Exception as e:
        log.debug(f"Image OCR error: {e}")
        return ""
    finally:
        if tmp_img and os.path.exists(tmp_img):
            os.unlink(tmp_img)

def extract_bytes(raw: bytes, ftype: str) -> str:
    if ftype == "pdf":                          return extract_pdf(raw)
    if ftype in ("docx", "doc"):               return extract_docx(raw)
    if ftype == "txt":                          return extract_txt(raw)
    if ftype in ("png", "jpg", "jpeg", "gif"): return extract_image(raw, ftype)
    return ""  # XLS, inconnu : classification par libellé uniquement

ARCHIVE_EXTRACTABLE = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"}

def _process_archive_members(members_iter, read_fn, type_name: str) -> list[tuple[str, str]]:
    results = []
    for m in members_iter:
        name_l = m.filename.lower()
        ext = os.path.splitext(name_l)[1]
        if ext not in ARCHIVE_EXTRACTABLE:
            continue
        if m.file_size > CONFIG.archive_member_max:
            log.warning(f"  {type_name} : {m.filename} ignoré (trop grand)")
            continue
        ftype = ext.lstrip(".")
        if ftype == "doc":
            ftype = "docx"
        text = extract_bytes(read_fn(m.filename), ftype)
        if text.strip():
            results.append((m.filename, text))
    return results
 
def extract_zip(raw: bytes) -> list[tuple[str, str]]:
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            log.info(f"  ZIP : {len(members)} membre(s)")
            return _process_archive_members(members, zf.read, "ZIP")
    except zipfile.BadZipFile as e:
        log.warning(f"  ZIP invalide : {e}"); return []
 
def extract_rar(raw: bytes) -> list[tuple[str, str]]:
    tmp_rar = None
    tmp_dir = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
            tmp.write(raw)
            tmp_rar = tmp.name

        tmp_dir = tempfile.mkdtemp()
        result = subprocess.run(
            ["unrar", "e", tmp_rar, tmp_dir + "/", "-y"],
            capture_output=True, text=True, timeout=30,
        )
        # unrar exit code 1 = avertissement non fatal (toujours OK)
        if result.returncode > 1:
            log.warning(f"  RAR unrar erreur : {result.stderr.strip()}")
            return []

        results = []
        for fname in sorted(os.listdir(tmp_dir)):
            fname_l = fname.lower()
            ext = os.path.splitext(fname_l)[1]
            if ext not in ARCHIVE_EXTRACTABLE:
                continue
            fpath = os.path.join(tmp_dir, fname)
            if os.path.getsize(fpath) > CONFIG.archive_member_max:
                log.warning(f"  RAR : {fname} ignoré (trop grand)")
                continue
            ftype = ext.lstrip(".")
            if ftype == "doc":
                ftype = "docx"
            with open(fpath, "rb") as f:
                text = extract_bytes(f.read(), ftype)
            if text.strip():
                results.append((fname, text))

        log.info(f"  RAR : {len(results)} fichier(s) extrait(s)")
        return results

    except Exception as e:
        log.warning(f"  RAR erreur : {e}"); return []
    finally:
        if tmp_rar and os.path.exists(tmp_rar):
            os.unlink(tmp_rar)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
 
 
# ─────────────────────── Téléchargement HTTP ─────────────────────
 
def download_file(url: str, declared_type: str) -> tuple[bytes, str]:
    is_archive = declared_type in ("zip", "rar")
 
    if is_archive:
        log.info("  Téléchargement complet (archive)...")
        resp = httpx.get(url, headers=HEADERS,
                         timeout=CONFIG.download_timeout, follow_redirects=True)
        resp.raise_for_status()
        ftype = detect_type(url, resp.headers.get("content-type", ""))
        return resp.content, ftype
 
    chunks, total = [], 0
    with httpx.stream("GET", url, headers=HEADERS,
                      timeout=CONFIG.download_timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        ftype = detect_type(url, resp.headers.get("content-type", "")) or declared_type
        for chunk in resp.iter_bytes(chunk_size=8192):
            chunks.append(chunk)
            total += len(chunk)
            if total >= CONFIG.stream_max_bytes:
                log.debug(f"  Limite {CONFIG.stream_max_bytes//1024} KB atteinte")
                break
    return b"".join(chunks), ftype