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
    return "inconnu"
