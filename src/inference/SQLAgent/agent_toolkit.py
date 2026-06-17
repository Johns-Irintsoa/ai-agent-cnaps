from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel


def get_toolkit(db: SQLDatabase, llm: BaseChatModel) -> SQLDatabaseToolkit:
    """Retourne un SQLDatabaseToolkit avec les 4 outils : list_tables, schema, query, query_checker."""
    return SQLDatabaseToolkit(db=db, llm=llm)
