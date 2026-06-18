-- Script d'initialisation Oracle XE pour le projet Admin RAG CNaPS
-- Exécuté lors du premier démarrage du conteneur Oracle (montage dans /opt/oracle/scripts/setup/)

-- Création de l'utilisateur dédié (connexion en SYS lors du setup)
CREATE USER RAG_ADMIN IDENTIFIED BY RAG;

GRANT CONNECT TO RAG_ADMIN;
GRANT RESOURCE TO RAG_ADMIN;
GRANT CREATE SESSION TO RAG_ADMIN;
GRANT CREATE TABLE TO RAG_ADMIN;
GRANT CREATE SEQUENCE TO RAG_ADMIN;
GRANT CREATE TRIGGER TO RAG_ADMIN;
ALTER USER RAG_ADMIN QUOTA UNLIMITED ON USERS;

-- Les tables RAG_DOCUMENT et RAG_DOCUMENT_LOG sont créées automatiquement
-- par Hibernate au démarrage de Spring Boot (ddl-auto: update).
-- Ce script ne crée que l'utilisateur et ses droits.
