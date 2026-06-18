"""
Client Oracle DB : SQLAlchemy engine + LangChain SQLDatabase.
Singleton avec tentative unique de connexion (gracieux si Oracle absent).
"""
import logging
import os
import time
from typing import Optional

from sqlalchemy import create_engine, text, Engine
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)

# Tables Oracle exposées au SQL Agent — frontière d'accès explicite
_ALLOWED_TABLES = ["sig_travailleurs", "cit2", "sig_mots_pas"]



try:
    import oracledb
    oracledb.init_oracle_client()
    logger.info("oracledb thick mode activé")
except Exception as _e:
    logger.info("oracledb thin mode (Instant Client absent ou déjà initialisé : %s)", _e)


class OracleClient:
    """Singleton pour la connexion Oracle (SQLAlchemy + LangChain SQLDatabase)."""

    ALLOWED_TABLES = _ALLOWED_TABLES

    _engine: Optional[Engine] = None
    _db: Optional[SQLDatabase] = None
    _initialized: bool = False

    @classmethod
    def get_engine(cls) -> Optional[Engine]:
        """Retourne le SQLAlchemy engine (tentative unique, None si Oracle inaccessible)."""
        if not cls._initialized:
            cls._initialized = True
            cls._engine = cls._try_connect()
        return cls._engine

    @classmethod
    def get_db(cls) -> Optional[SQLDatabase]:
        """Retourne le LangChain SQLDatabase limité au schéma ORACLE_SCHEMA."""
        if cls._db is None:
            engine = cls.get_engine()
            if engine is None:
                return None
            try:
                schema = os.getenv("ORACLE_SCHEMA")
                cls._db = SQLDatabase(engine, schema=schema, include_tables=_ALLOWED_TABLES)
                logger.info("SQLDatabase Oracle initialisé (schéma=%s)", schema)
            except Exception as e:
                logger.error("Impossible d'initialiser SQLDatabase Oracle : %s", e)
        return cls._db

    @classmethod
    def _try_connect(cls) -> Optional[Engine]:
        """Établit la connexion Oracle avec retry x3 et backoff. Retourne None si échec."""
        host = os.getenv("ORACLE_HOST")
        port = os.getenv("ORACLE_PORT")
        sid = os.getenv("ORACLE_SID")
        user = os.getenv("ORACLE_USER")
        password = os.getenv("ORACLE_PASSWORD")
        retries = 3
        backoff = 1.0

        conn_str = f"oracle+oracledb://{user}:{password}@{host}:{port}/{sid}"
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                engine = create_engine(conn_str, echo=False)
                with engine.connect():
                    pass
                logger.info("OracleClient connecté à %s:%s/%s", host, port, sid)
                return engine
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Tentative %d/%d échouée pour Oracle : %s. Nouvel essai dans %.1fs",
                    attempt + 1, retries, e, backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        logger.warning(
            "Oracle inaccessible après %d tentatives (%s). Fonctionnalité SQL désactivée.",
            retries, last_exc,
        )
        return None

    @classmethod
    def healthcheck(cls) -> bool:
        """Vérifie que Oracle est accessible."""
        try:
            engine = cls.get_engine()
            if engine is None:
                return False
            with engine.connect():
                pass
            return True
        except Exception:
            return False

    @classmethod
    def verify_auth(cls, matricule: str, password: str) -> bool:
        """Vérifie le couple matricule/mot de passe dans Oracle."""
        engine = cls.get_engine()
        if engine is None:
            return False
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT 1 FROM sig_mots_pas"
                        " WHERE travailleur_matricule = :m AND TRAVAILLEUR_PAS = :p"
                    ),
                    {"m": matricule, "p": password},
                ).fetchone()
                return row is not None
        except Exception as e:
            logger.error("Erreur verify_auth : %s", e)
            return False

    @classmethod
    def close(cls) -> None:
        """Réinitialise le singleton."""
        if cls._engine:
            cls._engine.dispose()
        cls._engine = None
        cls._db = None
        cls._initialized = False
        logger.info("OracleClient réinitialisé")
