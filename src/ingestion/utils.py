"""
Utils — Utilitaires d'ingestion de donnees CNaPS.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import os
from pathlib import Path
from typing import List


if TYPE_CHECKING:
    from ingestion.DataClasses import UrlCnapsWeb

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aplatissement de structures JSON imbriquees
# ---------------------------------------------------------------------------

def _flatten(obj: Any, prefix: str = "", sep: str = ".") -> dict:
    """
    Aplatit recursivement un dict ou une liste en un dict plat avec cles composees.

    Exemples :
        {"a": {"x": 1, "y": 2}, "b": 3}
            → {"a.x": 1, "a.y": 2, "b": 3}

        {"items": [{"id": 1}, {"id": 2}]}
            → {"items[0].id": 1, "items[1].id": 2}

    Args:
        obj:    Valeur a aplatir (dict, list ou scalaire).
        prefix: Prefixe de cle courant (utilise en recursion).
        sep:    Separateur entre les niveaux de cles (defaut : ".").

    Returns:
        Dict plat {cle_composee: valeur_scalaire}.
    """
    result: dict = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}{sep}{key}" if prefix else str(key)
            result.update(_flatten(value, full_key, sep))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full_key = f"{prefix}[{i}]"
            result.update(_flatten(item, full_key, sep))

    else:
        result[prefix] = obj

    return result


# ---------------------------------------------------------------------------
# Conversion JSON → tableau N colonnes
# ---------------------------------------------------------------------------

def json_to_table(
    json_path: str,
    columns: Optional[list[str]] = None,
    flatten_nested: bool = True,
    include_header: bool = True,
    fill_missing: Any = None,
) -> list[list]:
    """
    Convertit un fichier JSON en tableau a N colonnes (liste de listes).

    Le JSON peut etre :
      - Une liste d'objets  : [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
      - Un dict avec une cle contenant la liste : {"data": [...], "meta": ...}
      - Un objet unique     : {"a": 1, "b": 2}  → traite comme une liste a 1 ligne

    N (nombre de colonnes) est determine par :
      1. Le parametre 'columns' si fourni (N = len(columns))
      2. Sinon, l'union de toutes les cles des objets de la liste

    Args:
        json_path:      Chemin vers le fichier JSON.
        columns:        Liste optionnelle des colonnes a extraire (definit N).
                        Si None, toutes les cles trouvees sont utilisees.
                        Exemple : ["categorie", "confiance", "file_label"]
        flatten_nested: Si True (defaut), aplatit les objets imbriques avec
                        la notation pointee. Ex: {"a": {"x": 1}} → {"a.x": 1}
                        Si False, les valeurs imbriquees sont gardees telles quelles.
        include_header: Si True (defaut), la premiere ligne du tableau contient
                        les noms de colonnes.
        fill_missing:   Valeur utilisee pour les colonnes absentes dans une ligne
                        (defaut : None).

    Returns:
        list[list] — Tableau a N colonnes.
        Si include_header=True : table[0] = noms de colonnes, table[1:] = donnees.
        Si include_header=False : table[0] = premiere ligne de donnees.

    Exemples :
        # JSON : [{"nom": "Alice", "age": 30}, {"nom": "Bob", "age": 25}]
        table = json_to_table("data.json")
        # → [["nom", "age"], ["Alice", 30], ["Bob", 25]]

        # Extraire seulement 2 colonnes (N=2)
        table = json_to_table("data.json", columns=["nom", "age"])
        # → [["nom", "age"], ["Alice", 30], ["Bob", 25]]

        # JSON imbrique aplati :
        # [{"auteur": {"nom": "Alice"}, "score": 0.9}]
        table = json_to_table("data.json")
        # → [["auteur.nom", "score"], ["Alice", 0.9]]

    Raises:
        FileNotFoundError: Si json_path n'existe pas.
        ValueError:        Si le JSON ne contient pas de liste exploitable.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normaliser en liste de lignes
    rows_raw = _to_row_list(data, json_path)

    if not rows_raw:
        log.warning(f"json_to_table : aucune ligne dans {json_path}")
        return [columns] if (include_header and columns) else []

    # Aplatir les objets imbriques si demande
    if flatten_nested:
        rows_flat = [_flatten(row) if isinstance(row, dict) else {"[0]": row}
                     for row in rows_raw]
    else:
        rows_flat = [row if isinstance(row, dict) else {"[0]": row}
                     for row in rows_raw]

    # Determiner les colonnes (N)
    if columns is None:
        # Union ordonnee de toutes les cles rencontrees
        seen: dict = {}
        for row in rows_flat:
            for key in row:
                seen[key] = None
        columns = list(seen.keys())

    n = len(columns)
    log.info(f"json_to_table : {len(rows_flat)} ligne(s) × {n} colonne(s) depuis {path.name}")

    # Construire le tableau
    table: list[list] = []

    if include_header:
        table.append(columns)

    for row in rows_flat:
        table.append([row.get(col, fill_missing) for col in columns])

    return table


# ---------------------------------------------------------------------------
# Utilitaire interne : normaliser n'importe quel JSON en liste de lignes
# ---------------------------------------------------------------------------

def _to_row_list(data: Any, source: str = "") -> list:
    """
    Convertit une valeur JSON arbitraire en liste de lignes exploitables.

    Regles :
      - list  → retournee directement
      - dict  → si une seule valeur est une liste, c'est la liste de donnees ;
                 sinon le dict entier est traite comme une seule ligne
      - autre → encapsule dans [data]

    Args:
        data:   Valeur JSON deja parsee.
        source: Chemin source (pour les logs).

    Returns:
        Liste de lignes (chaque element sera un dict ou une valeur scalaire).
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Chercher la premiere valeur de type list (ex: {"data": [...], "total": 5})
        list_values = [(k, v) for k, v in data.items() if isinstance(v, list)]
        if len(list_values) == 1:
            key, lst = list_values[0]
            log.debug(f"  _to_row_list : utilisation de la cle '{key}' comme source de lignes")
            return lst
        if len(list_values) > 1:
            # Plusieurs listes : prendre la plus grande
            key, lst = max(list_values, key=lambda kv: len(kv[1]))
            log.debug(f"  _to_row_list : plusieurs listes trouvees, utilisation de '{key}' ({len(lst)} lignes)")
            return lst
        # Pas de liste dans le dict → traiter le dict comme une seule ligne
        return [data]

    # Scalaire ou autre → une seule ligne
    return [data]


# ---------------------------------------------------------------------------
# Conversion cnaps_urls.json → liste de UrlCnapsWeb
# ---------------------------------------------------------------------------

def convert_json_to_list(json_path: str) -> "list[UrlCnapsWeb]":
    """
    Lit le fichier cnaps_urls.json et retourne la liste des entrees sous forme
    d'objets UrlCnapsWeb.

    Args:
        json_path: Chemin vers le fichier JSON (ex: "cnaps_urls.json").

    Returns:
        list[UrlCnapsWeb] — Un objet par entree dans le tableau "cnaps_urls".

    Raises:
        FileNotFoundError: Si json_path n'existe pas.
    """
    from ingestion.DataClasses import UrlCnapsWeb

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("cnaps_urls", [])
    log.info(f"convert_json_to_list : {len(entries)} URL(s) trouvees dans {path.name}")
    return [UrlCnapsWeb(url=e["url"], attrClasses=e.get("classes", [])) for e in entries]


