import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, OllamaEmbeddingFunction
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, Collection, Visibility

DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data"
SQLITE_DB_PATH = DATA_DIR / "lamb-kb-server.db"
CHROMA_DB_PATH = DATA_DIR / "chromadb"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DB_PATH),
    settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)


def _get_embedding_function_by_params(vendor: str, model_name: str, api_key: str = "", api_endpoint: str = ""):
    vendor = vendor.lower()
    
    if vendor in ("ollama", "local"):
        return OllamaEmbeddingFunction(
            url=api_endpoint or "http://localhost:11434",
            model_name=model_name
        )
    
    elif vendor == "openai":
        kwargs = {"api_key": api_key, "model_name": model_name}
        if api_endpoint:
            kwargs["api_base"] = api_endpoint
        return OpenAIEmbeddingFunction(**kwargs)
    
    else:
        return None


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_chroma_client() -> chromadb.PersistentClient:
    return chroma_client


def init_sqlite_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_embedding_function(collection_id_or_obj: Union[int, Collection, Dict[str, Any]]) -> Callable:
    db = next(get_db())
    
    # Handle dict case first
    if isinstance(collection_id_or_obj, dict):
        if 'embeddings_model' in collection_id_or_obj:
            embedding_config = collection_id_or_obj['embeddings_model']
            db.close()
            return _get_embedding_function_by_params(
                embedding_config.get("vendor"),
                embedding_config.get("model"),
                embedding_config.get("apikey"),
                embedding_config.get("api_endpoint")
            )
        collection_id = collection_id_or_obj.get('id')
        if not collection_id:
            db.close()
            return None
        collection = db.query(Collection).filter(Collection.id == collection_id).first()
    # Handle Collection object case
    elif isinstance(collection_id_or_obj, Collection):
        collection = collection_id_or_obj
    # Handle integer ID case
    elif isinstance(collection_id_or_obj, int):
        collection = db.query(Collection).filter(Collection.id == collection_id_or_obj).first()
    else:
        db.close()
        return None
        
    if not collection:
        db.close()
        return None
        
    # Extract embedding configuration
    embedding_config = json.loads(collection.embeddings_model) if isinstance(collection.embeddings_model, str) else collection.embeddings_model
    
    # Use the helper function to get the actual embedding function
    result = _get_embedding_function_by_params(
        embedding_config.get("vendor"),
        embedding_config.get("model"),
        embedding_config.get("apikey"),
        embedding_config.get("api_endpoint")
    )
    
    db.close()
    return result


def _check_sqlite_schema() -> bool:
    inspector = inspect(engine)
    
    if "collections" not in inspector.get_table_names():
        return True
    
    collection_columns = {col["name"] for col in inspector.get_columns("collections")}
    required_columns = {"id", "name", "description", "creation_date", "owner", "visibility", "embeddings_model"}
    
    return required_columns.issubset(collection_columns)


def init_databases() -> Dict[str, Any]:
    status = {
        "sqlite_initialized": False,
        "sqlite_schema_valid": False,
        "chromadb_initialized": False,
        "errors": []
    }
    
    status["sqlite_schema_valid"] = _check_sqlite_schema()
    
    if not status["sqlite_schema_valid"]:
        status["errors"].append("SQLite schema is not compatible")
    
    init_sqlite_db()
    status["sqlite_initialized"] = True
    
    status["chromadb_collections"] = len(chroma_client.list_collections())
    status["chromadb_initialized"] = True
    
    return status