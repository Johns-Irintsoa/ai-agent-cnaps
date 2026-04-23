from pypdf import PdfReader
from docx import Document as DocxDocument
from io import BytesIO
import os, zipfile, rarfile, httpx, logging, tempfile, subprocess, shutil
from data_classificateur.config import CONFIG
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
            capture_output=True, text=False, timeout=30,
        )
        # unrar exit code 1 = avertissement non fatal (toujours OK)
        if result.returncode > 1:
            stderr = result.stderr.decode("latin-1", errors="replace").strip()
            log.warning(f"  RAR unrar erreur : {stderr}")
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
 
 
def extract_zip_bytes(raw: bytes) -> list[tuple[str, bytes]]:
    """Extrait tous les membres d'une archive ZIP et retourne leurs bytes bruts."""
    results = []
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                if m.file_size > CONFIG.archive_member_max:
                    log.warning(f"  ZIP : {m.filename} ignoré (trop grand)")
                    continue
                results.append((m.filename, zf.read(m.filename)))
        log.info(f"  ZIP : {len(results)} membre(s) extrait(s)")
    except zipfile.BadZipFile as e:
        log.warning(f"  ZIP invalide : {e}")
    return results


def extract_rar_bytes(raw: bytes) -> list[tuple[str, bytes]]:
    """Extrait tous les membres d'une archive RAR et retourne leurs bytes bruts."""
    tmp_rar = None
    tmp_dir = None
    results = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
            tmp.write(raw)
            tmp_rar = tmp.name
        tmp_dir = tempfile.mkdtemp()
        result = subprocess.run(
            ["unrar", "e", tmp_rar, tmp_dir + "/", "-y"],
            capture_output=True, text=False, timeout=30,
        )
        stdout = result.stdout.decode("latin-1", errors="replace").strip()
        stderr = result.stderr.decode("latin-1", errors="replace").strip()
        log.info(f"  RAR unrar returncode={result.returncode}")
        if stdout:
            log.info(f"  RAR unrar stdout: {stdout[:500]}")
        if stderr:
            log.info(f"  RAR unrar stderr: {stderr[:200]}")
        if result.returncode > 1:
            log.warning(f"  RAR unrar erreur fatale")
            return []
        files_in_dir = os.listdir(tmp_dir)
        log.info(f"  RAR tmp_dir contient : {files_in_dir}")
        for fname in sorted(files_in_dir):
            fpath = os.path.join(tmp_dir, fname)
            if os.path.getsize(fpath) > CONFIG.archive_member_max:
                log.warning(f"  RAR : {fname} ignoré (trop grand)")
                continue
            with open(fpath, "rb") as f:
                results.append((fname, f.read()))
        log.info(f"  RAR : {len(results)} membre(s) extrait(s)")
    except Exception as e:
        log.warning(f"  RAR erreur : {e}")
    finally:
        if tmp_rar and os.path.exists(tmp_rar):
            os.unlink(tmp_rar)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


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
 
    resp = httpx.get(url, headers=HEADERS,
                     timeout=CONFIG.download_timeout, follow_redirects=True)
    resp.raise_for_status()
    ftype = detect_type(url, resp.headers.get("content-type", "")) or declared_type
    return resp.content, ftype

def detect_type(url: str, ct: str) -> str:
    u, c = url.lower().split("?")[0], ct.lower()
    if u.endswith(".pdf") or "application/pdf" in c:   return "pdf"
    if u.endswith(".docx") or "wordprocessingml" in c: return "docx"
    if u.endswith(".doc") or "msword" in c:            return "doc"
    if u.endswith(".xls") or "ms-excel" in c:          return "xls"
    if u.endswith(".xlsx") or "spreadsheetml" in c:    return "xlsx"
    if u.endswith(".zip") or "zip" in c:               return "zip"
    if u.endswith(".rar") or "rar" in c:               return "rar"
    if u.endswith(".txt") or "text/plain" in c:        return "txt"
    if u.endswith(".png") or "image/png" in c:         return "png"
    if u.endswith(".jpg") or u.endswith(".jpeg") or "image/jpeg" in c: return "jpg"
    if u.endswith(".gif") or "image/gif" in c:         return "gif"
    if u.endswith(".webp") or "image/webp" in c:       return "webp"
    return "inconnu"
