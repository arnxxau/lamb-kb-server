"""
Database connection module for SQLite and ChromaDB.

This module provides connection functions for both SQLite (via SQLAlchemy) and ChromaDB.
It handles initialization, connection management, and basic sanity checks.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, Collection, Visibility

# Database paths
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data"
SQLITE_DB_PATH = DATA_DIR / "lamb-kb-server.db"
CHROMA_DB_PATH = DATA_DIR / "chromadb"

# Ensure the directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)

# Create SQLite engine
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DB_PATH),
    settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)


def get_embedding_function(vendor: str, model_name: str, api_key: Optional[str] = None) -> Union[Callable, None]:
    """Get the appropriate embedding function based on vendor and model.
    
    Args:
        vendor: The embedding vendor ('local', 'openai', etc.)
        model_name: The name of the embedding model
        api_key: API key for the vendor (if required)
        
    Returns:
        An embedding function compatible with ChromaDB or None for default
        
    Raises:
        ValueError: If an invalid vendor is specified or required API key is missing
    """
    print(f"DEBUG: [get_embedding_function] Getting embedding function for vendor: {vendor}, model: {model_name}")
    
    # Handle None vendor or empty string - default to local
    if vendor is None or vendor == "":
        vendor = "local"
        print(f"DEBUG: [get_embedding_function] Using default vendor: {vendor}")
        
    if vendor.lower() == 'local':
        # For local embeddings, ChromaDB will use the default embedding function
        # based on the model_name when creating a collection
        print(f"DEBUG: [get_embedding_function] Using local embeddings with model: {model_name}")
        print(f"DEBUG: [get_embedding_function] Returning None to let ChromaDB use default embedding function")
        return None
    
    elif vendor.lower() == 'openai':
        print(f"DEBUG: [get_embedding_function] Using OpenAI embeddings with model: {model_name}")
        if not api_key:
            print(f"DEBUG: [get_embedding_function] ERROR: Missing API key for OpenAI")
            raise ValueError("API key is required for OpenAI embeddings")
        
        # Mask the API key for logging
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"  
        print(f"DEBUG: [get_embedding_function] Using OpenAI API key: {masked_key}")
        
        # Add timeout handling for OpenAI API
        try:
            print(f"DEBUG: [get_embedding_function] Creating OpenAI embedding function")
            # Create a custom wrapper around the OpenAI embedding function
            # to add better error handling and timeout management
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name,
                # Add longer timeout to prevent hanging
                # OpenAI's default is 600s but can be reduced
                timeout=60  # Set a 60 second timeout to prevent indefinite hanging
            )
            print(f"DEBUG: [get_embedding_function] Successfully created OpenAI embedding function")
            
            # Create a wrapper function that adds debugging
            def debug_embedding_function(texts):
                print(f"DEBUG: [embedding_function] Generating embeddings for {len(texts)} texts")
                print(f"DEBUG: [embedding_function] First text sample: {texts[0][:100]}...")
                try:
                    import time
                    start_time = time.time()
                    print(f"DEBUG: [embedding_function] Calling OpenAI API")
                    result = openai_ef(texts)
                    end_time = time.time()
                    print(f"DEBUG: [embedding_function] OpenAI API call completed in {end_time - start_time:.2f} seconds")
                    return result
                except Exception as e:
                    print(f"DEBUG: [embedding_function] ERROR calling OpenAI API: {str(e)}")
                    import traceback
                    print(f"DEBUG: [embedding_function] Stack trace:\n{traceback.format_exc()}")
                    raise
            
            return debug_embedding_function
            
        except Exception as e:
            print(f"DEBUG: [get_embedding_function] ERROR creating OpenAI embedding function: {str(e)}")
            import traceback
            print(f"DEBUG: [get_embedding_function] Stack trace:\n{traceback.format_exc()}")
            raise
    
    else:
        print(f"DEBUG: [get_embedding_function] ERROR: Unsupported embedding vendor: {vendor}")
        raise ValueError(f"Unsupported embedding vendor: {vendor}")


def get_db() -> Session:
    """Get a database session.
    
    Returns:
        A SQLAlchemy Session object
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_chroma_client() -> chromadb.PersistentClient:
    """Get the ChromaDB client.
    
    Returns:
        The ChromaDB client instance
    """
    print(f"DEBUG: [get_chroma_client] Returning ChromaDB client with path: {CHROMA_DB_PATH}")
    return chroma_client


def init_sqlite_db() -> None:
    """Initialize the SQLite database.
    
    Creates all tables if they don't exist.
    """
    Base.metadata.create_all(bind=engine)
    print(f"SQLite database initialized at: {SQLITE_DB_PATH}")


def check_sqlite_schema() -> bool:
    """Check if the SQLite database schema is compatible.
    
    Returns:
        True if schema is valid, False otherwise
    """
    inspector = inspect(engine)
    
    # Check if collections table exists
    if "collections" not in inspector.get_table_names():
        print("Collections table not found in database. Will be created.")
        return True
    
    # Check if collections table has all required columns
    collection_columns = {col["name"] for col in inspector.get_columns("collections")}
    required_columns = {"id", "name", "description", "creation_date", "owner", "visibility", "embeddings_model"}
    
    if not required_columns.issubset(collection_columns):
        missing = required_columns - collection_columns
        print(f"Missing columns in collections table: {missing}")
        return False
    
    return True


def init_databases() -> Dict[str, Any]:
    """Initialize all databases and perform sanity checks.
    
    Returns:
        Dict with initialization status information
    """
    status = {
        "sqlite_initialized": False,
        "sqlite_schema_valid": False,
        "chromadb_initialized": False,
        "errors": []
    }
    
    try:
        # Check SQLite schema before initializing
        schema_valid = check_sqlite_schema()
        status["sqlite_schema_valid"] = schema_valid
        
        if not schema_valid:
            status["errors"].append("SQLite schema is not compatible")
        
        # Initialize SQLite database
        init_sqlite_db()
        status["sqlite_initialized"] = True
        
        # Test ChromaDB connection
        collections = chroma_client.list_collections()
        status["chromadb_initialized"] = True
        status["chromadb_collections"] = len(collections)
        
    except Exception as e:
        status["errors"].append(f"Error initializing databases: {str(e)}")
    
    return status
