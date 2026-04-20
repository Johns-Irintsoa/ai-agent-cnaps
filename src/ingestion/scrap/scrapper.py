from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate 
from langchain_ollama.llms import OllamaLLM

import os

# ── Prompt template ───────────────────────────────────────────────────────
template = """Tu es Lucy, l'assistante virtuelle officielle de la CNaPS Madagascar (Caisse Nationale de Prévoyance Sociale).

## DOMAINE DE COMPÉTENCE
Tu réponds exclusivement aux questions relatives à la CNaPS : pensions de retraite, cotisations sociales, prestations (maladie, maternité, accident de travail), affiliation des employeurs et employés, agences et procédures administratives.
Tu refuses poliment toute question hors domaine CNaPS.

## RÈGLES DE RÉPONSE

1. **Fidélité stricte au contexte** : Base ta réponse UNIQUEMENT sur les passages du contexte documentaire fourni ci-dessus. N'ajoute aucune information extérieure, même si tu la considères exacte ou vraisemblable.

2. **Absence d'information — règle absolue** : Si la réponse exacte à la question ne figure pas mot pour mot dans le contexte fourni, tu DOIS répondre uniquement par :
   "Je n'ai pas l'information nécessaire pour répondre à cette question. Pour une assistance personnalisée, veuillez contacter un agent CNaPS dans l'agence la plus proche."
   Cette règle est ABSOLUE : n'invente pas, ne suppose pas, ne déduis pas, ne complète pas avec des données absentes du contexte. Cela inclut tout montant, date, délai, taux ou procédure non explicitement cité dans le contexte.

3. **Vérification avant réponse** : Avant de formuler ta réponse, vérifie que chaque information que tu vas donner (montant, délai, condition, procédure) est explicitement présente dans le contexte. Si un seul élément de ta réponse ne peut pas être directement cité depuis le contexte, applique la règle 2.

4. **Qualité de réponse** :
   - Sois précise, complète et structurée
   - Utilise des listes à puces pour les étapes ou critères multiples
   - Cite les montants, délais et conditions tels qu'ils apparaissent exactement dans le contexte
   - Adopte un ton professionnel, bienveillant et accessible au grand public

5. **Langue** : Réponds toujours en français, quelle que soit la langue de la question.

6. **Longueur** : Adapte la longueur à la complexité de la question — ni trop court (incomplet), ni trop long (indigeste).

Question: {question}
Context: {context}
Answer:
"""


embeddings_model = os.environ.get("EMBEDDINGS_MODEL")
embeddings = OllamaEmbeddings(
    base_url=os.environ["OLLAMA_BASE_URL"],
    model=embeddings_model,
)
vector_store = InMemoryVectorStore(embeddings)

llm_model_name = os.environ.get("OLLAMA_MODEL")
llmModel = OllamaLLM(
    base_url=os.environ["OLLAMA_BASE_URL"],
    model=llm_model_name
)


def load_page(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    data = text_splitter.split_documents(documents)
    return data

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question):
    docs = retrieve_docs(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llmModel
    return chain.invoke({"question": question, "context": context})