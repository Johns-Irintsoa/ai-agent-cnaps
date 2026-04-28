
def ingestion_url_process (url: str) -> list[Document]:
    """
    Traite une URL specifique en utilisant load_html_from_url.

    Args:
        url: L'URL a traiter.

    Returns:
        Liste de Document LangChain extraits de l'URL.
    """
    log.info(f"ingestion_url_process : Traitement de l'URL {url}")
    docs = load_html_from_url(url)
    log.info(f"ingestion_url_process : {len(docs)} document(s) extraits de l'URL")
    return docs

def ingestion_web_data() -> dict:
    """
    Traite toutes les URLs definies dans cnaps_urls.json en utilisant load_web_data.
    Traite touts les fichiers qui ont ete classifies comme "utile" par le pipeline de filtrage.

    Returns:
        Dictionnaire de donnees dans laquelle liste tous les documents charges et traites, avec un statut global de l'operation.
        Dictionnaire avec le statut et le message de l'operation.
    """
