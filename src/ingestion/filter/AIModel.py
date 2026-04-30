import textwrap

from pydantic import BaseModel, Field
from LLM.llm import LLMClient 

# Définition du schéma de sortie
class RAGDecision(BaseModel):
    categorie: str = Field(description="Type de document : FORMULAIRE, TABLEAU, TEXTE ou AUTRE")
    confiance: float = Field(description="Score de confiance entre 0.0 et 1.0")
    raison: str = Field(description="Explication courte en une phrase")

    @property
    def category(self) -> str:
        return self.categorie
    
    @property
    def reason(self) -> str:
        return self.raison

    @property
    def is_useful(self) -> bool:
        return self.categorie.upper() != "AUTRE"

class DocumentValidator:
    def __init__(self, client: LLMClient):
        # On lie le schéma Pydantic au modèle via la propriété .model
        self.structured_llm = client.model.with_structured_output(RAGDecision)

    def validate(self, content_sample: str, filename: str) -> RAGDecision:
        group = "unknown"
        body = content_sample.strip()[:2000] if content_sample.strip() else "(texte non extractible)"
        prompt = textwrap.dedent(f'''Tu es un classificateur universel de documents. Ton role est
    d'analyser le contenu d'un document, quelle que soit sa langue,
    son domaine, son format ou son origine, et de le classer dans
    l'une des 4 categories suivantes :
    FORMULAIRE, TABLEAU, TEXTE ou AUTRE.

    ## DEFINITIONS FONCTIONNELLES

    FORMULAIRE : document de recolte d'information.
      Structure a zones d'attente destinees a completion
      par une tierce personne. Peut etre statique (zones vides)
      ou actif (calculs declenches par la saisie).

    TABLEAU : document de transmission de donnees structurees
      et figees. Organisation matricielle ligne x colonne
      avec en-tetes explicites. Donnees completes, mesurables,
      accompagnees d'agregations ou de visualisations.

    TEXTE : document de transmission d'information narrative,
      prescriptive ou argumentative, destine a etre lu de maniere
      continue ou consulte. Contenu complet, structure par la
      langue. Produit par un auteur ou une institution identifiable.
      Intention : communiquer, informer, instruire, raconter,
      guider ou convaincre un lecteur.

    AUTRE : document ne relevant d'aucune des trois categories
      precedentes. Contenu inclassable ou mixte ne presentant
      pas suffisamment de signaux pour une categorie definie.

    ────────────────────────────────────────────────────────────
    ## GRILLE D'ANALYSE — CRITERES PAR CATEGORIE
    ────────────────────────────────────────────────────────────

    ### [FORMULAIRE] — Signaux structurels

    NIVEAU 1 — Directs (3 pts chacun)
    [C1.1] ZONES D'ATTENTE VIDES
      Espaces de saisie future : lignes vides repetees,
      sequences de points post-label, cases delimitees,
      blancs post-":", cellules vides en tableau,
      colonnes structurees sans donnees.

    [C1.2] INSTRUCTION AU REMPLISSSEUR
      Adressage direct au completeur : injonction de
      remplissage, consigne de format (majuscules, encre
      noire, format date), instruction d'impression ou
      d'envoi post-completion, avertissement de rejet
      en cas de non-conformite.

    [C1.3] CHOIX MUTUELLEMENT EXCLUSIFS
      Options a selection unique : cases a cocher ou
      a croix, options a rayer, boutons radio imprimes,
      listes deroulantes PDF ou tableur interactif.

    [C1.4] LOGIQUE DE CALCUL CONDITIONNEE A LA SAISIE
      Formules, cellules calculees ou totaux automatiques
      dont le resultat depend de donnees a saisir.
      Signal fort : structure multi-feuilles ou une feuille
      agrege les resultats des autres.

    NIVEAU 2 — Forts (2 pts chacun)
    [C2.1] EMPLACEMENTS DE VALIDATION MULTIPLES
      Zones de signature, paraphe ou cachet pour parties
      distinctes — circulation multi-intervenants
      a validations successives.

    [C2.2] IDENTIFIANT DE MODELE NORMALISE
      Code ou reference d'identification du gabarit
      lui-meme (non du contenu) : numero de formulaire,
      code de version, reference reglementaire du modele.

    [C2.3] SECTION RESERVEE A TIERS NON-REMPLISSEUR
      Delimitation explicite d'une zone inaccessible
      au deposant : reservation a administration,
      autorite de controle ou toute entite tierce.

    [C2.4] CHAMPS D'IDENTIFIANTS EN CASES UNITAIRES
      Saisie caractere par caractere d'identifiants
      en cases individuelles de taille fixe — contrainte
      de format garantie par la structure.

    [C2.5] REGLES DE VALIDATION STRUCTURELLE INTEGREES
      Contraintes documentees dans le gabarit : unicite
      d'entree par entite, interdiction de modifier la
      structure, format impose par champ, marquage
      visuel des champs obligatoires.

    NIVEAU 3 — Contextuels (1 pt chacun)
    [C3.1] TITRE ORIENTE ACTION OU REQUETE
    [C3.2] STRUCTURE SYMETRIQUE MULTILINGUE
    [C3.3] NOTICE D'AIDE AU REMPLISSAGE
    [C3.4] ENGAGEMENT FORMEL DU SIGNATAIRE
    [C3.5] MULTIPLICITE DES PARTIES IMPLIQUEES

    ────────────────────────────────────────────────────────────

    ### [TABLEAU] — Signaux structurels

    NIVEAU 1 — Directs (3 pts chacun)
    [T1.1] COMPLETUDE TOTALE DES CELLULES
    [T1.2] STRUCTURE LIGNE x COLONNE A EN-TETES EXPLICITES
    [T1.3] AGREGATION VERIFIABLE
    [T1.4] ATTRIBUTION A UN EMETTEUR DATE

    NIVEAU 2 — Forts (2 pts chacun)
    [T2.1] VISUALISATIONS DERIVEES DES DONNEES
    [T2.2] COLONNES DE CALCUL DERIVE FIGEES
    [T2.3] STRUCTURE MULTI-SECTIONS THEMATIQUES COMPLETES

    NIVEAU 3 — Contextuels (1 pt chacun)
    [T3.1] TITRE DESCRIPTIF DE CONTENU
    [T3.2] ABSENCE D'INSTRUCTION AU LECTEUR
    [T3.3] ABSENCE D'ENGAGEMENT FORMEL
    [T3.4] COMPARAISON TEMPORELLE EXPLICITE
    [T3.5] CROISEMENT DE DEUX AXES CATEGORIELS

    ────────────────────────────────────────────────────────────

    ### [TEXTE] — Signaux structurels

    NIVEAU 1 — Directs (3 pts chacun)
    [X1.1] CONTENU NARRATIF OU DISCURSIF CONTINU
    [X1.2] STRUCTURE EDITORIALE OU DOCUMENTAIRE
    [X1.3] AUTEUR OU EMETTEUR IDENTIFIE

    NIVEAU 2 — Forts (2 pts chacun)
    [X2.1] DESTINATION LECTURE OU CONSULTATION
    [X2.2] CONTENU PRESCRIPTIF SEQUENTIEL
    [X2.3] CONTENU ARGUMENTATIF OU INTERPRETATIF
    [X2.4] CONTENU REFERENTIEL CONSULTATIF

    NIVEAU 3 — Contextuels (1 pt chacun)
    [X3.1] TITRE INFORMATIF OU PRESCRIPTIF
    [X3.2] PRESENCE DE VISUELS ILLUSTRATIFS
    [X3.3] PERIODICITE OU VERSIONNAGE
    [X3.4] MULTILINGUISME EDITORIAL
    [X3.5] ADRESSE DIRECTE AU LECTEUR

    ────────────────────────────────────────────────────────────
    ## ARBRE DE DECISION
    ────────────────────────────────────────────────────────────

    ETAPE A — TEST FORMULAIRE : score C1-C3 >= 4 -> FORMULAIRE
    ETAPE B — TEST TABLEAU    : score T1-T3 >= 4 -> TABLEAU
    ETAPE C — TEST TEXTE      : score X1-X3 >= 4 -> TEXTE
    ETAPE D — FALLBACK        : aucun test concluant -> AUTRE

    ────────────────────────────────────────────────────────────
    ## DOCUMENT A ANALYSER
    ────────────────────────────────────────────────────────────

    Contexte :
    - Groupe de la page : "{group}"
    - Libelle du fichier : "{filename}"

    Extrait du document :
    """
    {body}
    """

    Reponds UNIQUEMENT en JSON valide :
    {{
      "categorie": "<FORMULAIRE | TABLEAU | TEXTE | AUTRE>",
      "confiance": <0.0 a 1.0>,
      "raison": "<explication courte en 1 phrase>"
    }}''')

        return self.structured_llm.invoke(prompt)