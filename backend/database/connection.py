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


def get_embedding_function_by_params(vendor: str, model_name: str, api_key: Optional[str] = None, api_endpoint: Optional[str] = None) -> Union[Callable, None]:
    """Get the appropriate embedding function based on explicit parameters.
    This is an internal helper function used by get_embedding_function.
    
    Args:
        vendor: The embedding vendor ('local', 'openai', 'ollama', etc.)
        model_name: The name of the embedding model
        api_key: API key for the vendor (if required)
        api_endpoint: Endpoint URL for the API (if applicable)
        
    Returns:
        An embedding function compatible with ChromaDB
        
    Raises:
        ValueError: If an invalid vendor is specified or required parameters are missing
    """
    print(f"DEBUG: [get_embedding_function_by_params] Getting embedding function for vendor: {vendor}, model: {model_name}")
    
    # Validate that vendor is provided
    if vendor is None or vendor == "":
        # Set Ollama as the default vendor
        vendor = "ollama"
        print(f"DEBUG: [get_embedding_function_by_params] No vendor specified, defaulting to Ollama")
        
    if vendor.lower() == 'local':
        # Local embedding is not recommended due to dependency issues
        print(f"DEBUG: [get_embedding_function_by_params] WARNING: Local embeddings not recommended, switching to Ollama")
        print(f"DEBUG: [get_embedding_function_by_params] To use local embeddings, explicitly install sentence-transformers package")
        
        # Try to fall back to Ollama 
        import os
        vendor = "ollama"
        model_name = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
        api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT", "http://localhost:11434/api/embeddings")
        print(f"DEBUG: [get_embedding_function_by_params] Falling back to Ollama with model: {model_name}")
        
        # Continue to next vendor check (will hit the Ollama case)
    
    elif vendor.lower() == 'ollama' or vendor.lower() == 'ollama-local':
        print(f"DEBUG: [get_embedding_function_by_params] Using Ollama embeddings with model: {model_name}")
        import requests
        import os
        
        # Get the endpoint URL from parameters or environment
        # When loading from collection config, api_key might actually be a JSON dict with additional params
        endpoint = None
        if isinstance(api_key, dict) and 'endpoint' in api_key:
            endpoint = api_key['endpoint']
            print(f"DEBUG: [get_embedding_function_by_params] Using endpoint from collection config: {endpoint}")
        else:
            # Use provided api_endpoint parameter if available
            endpoint = api_endpoint
            if not endpoint:
                # Default to environment variable with no hardcoded fallback
                endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
                if not endpoint:
                    print(f"DEBUG: [get_embedding_function_by_params] ERROR: No API endpoint specified for Ollama")
                    raise ValueError("No API endpoint specified for Ollama embeddings. Please set EMBEDDINGS_ENDPOINT environment variable.")
                print(f"DEBUG: [get_embedding_function_by_params] Using endpoint from environment: {endpoint}")
        
        # Ensure endpoint ends with /api/embeddings
        if not endpoint.endswith('/embeddings'):
            if endpoint.endswith('/api'):
                endpoint += '/embeddings'
            elif not endpoint.endswith('/api/'):
                endpoint += '/api/embeddings'
        
        # Test if Ollama service is available before proceeding
        base_url = endpoint.split('/api/')[0]
        try:
            print(f"DEBUG: [get_embedding_function_by_params] Testing Ollama service at: {base_url}")
            response = requests.get(f"{base_url}/api/version", timeout=2)
            response.raise_for_status()
            print(f"DEBUG: [get_embedding_function_by_params] Ollama service is available: {response.json()}")
        except Exception as e:
            print(f"DEBUG: [get_embedding_function_by_params] ERROR: Ollama service is not available: {str(e)}")
            raise ValueError(f"Ollama service is not available at {base_url}. Error: {str(e)}")
        
        def ollama_embedding_function(texts):
            print(f"DEBUG: [embedding_function] Generating embeddings for {len(texts)} texts with Ollama")
            print(f"DEBUG: [embedding_function] Using model: {model_name} at endpoint: {endpoint}")
            results = []
            
            for text in texts:
                try:
                    response = requests.post(
                        endpoint,
                        json={
                            "model": model_name,
                            "prompt": text
                        },
                        timeout=30  # Add timeout to prevent hanging
                    )
                    response.raise_for_status()
                    embedding = response.json().get('embedding')
                    if embedding:
                        results.append(embedding)
                    else:
                        error_msg = f"No embedding returned from Ollama API: {response.json()}"
                        print(f"DEBUG: [embedding_function] ERROR: {error_msg}")
                        raise ValueError(error_msg)
                except Exception as e:
                    print(f"DEBUG: [embedding_function] ERROR calling Ollama API: {str(e)}")
                    raise
            
            return results
        
        return ollama_embedding_function
        
    elif vendor.lower() == 'openai':
        print(f"DEBUG: [get_embedding_function_by_params] Using OpenAI embeddings with model: {model_name}")
        if not api_key:
            print(f"DEBUG: [get_embedding_function_by_params] ERROR: Missing API key for OpenAI")
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
            print(f"DEBUG: [get_embedding_function_by_params] Successfully created OpenAI embedding function")
            
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
            print(f"DEBUG: [get_embedding_function_by_params] ERROR creating OpenAI embedding function: {str(e)}")
            import traceback
            print(f"DEBUG: [get_embedding_function_by_params] Stack trace:\n{traceback.format_exc()}")
            raise
    
    else:
        print(f"DEBUG: [get_embedding_function_by_params] ERROR: Unsupported embedding vendor: {vendor}")
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


def get_embedding_function(collection_id: int) -> Callable:
    """Get the embedding function for a collection by its ID.
    
    This function fetches the embedding configuration from the database
    and constructs the appropriate embedding function.
    
    Args:
        collection_id: ID of the collection
        
    Returns:
        An embedding function compatible with ChromaDB
        
    Raises:
        ValueError: If the collection is not found or has invalid configuration
        RuntimeError: If embeddings service is not available
    """
    from sqlalchemy.orm import Session
    from database.models import Collection
    import json
    import os
    
    print(f"DEBUG: [get_embedding_function] Getting embedding function for collection_id: {collection_id}")
    
    # Get SQLite database session
    db = next(get_db())
    
    try:
        # Get the collection from the database
        collection = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            print(f"DEBUG: [get_embedding_function] ERROR: Collection not found with ID: {collection_id}")
            raise ValueError(f"Collection not found with ID: {collection_id}")
            
        # Extract embedding configuration
        embedding_config = json.loads(collection.embeddings_model) if isinstance(collection.embeddings_model, str) else collection.embeddings_model
        
        # Get configuration parameters
        model = embedding_config.get("model")
        vendor = embedding_config.get("vendor")
        api_key = embedding_config.get("apikey")
        api_endpoint = embedding_config.get("api_endpoint")
        
        # Check if any values are still 'default' - this shouldn't happen if collection was created properly
        if model == "default" or vendor == "default" or api_key == "default" or api_endpoint == "default":
            print(f"WARNING: [get_embedding_function] Found 'default' values in collection {collection_id} configuration.")
            print(f"WARNING: [get_embedding_function] These should have been resolved during collection creation.")
            print(f"WARNING: [get_embedding_function] Current config: model={model}, vendor={vendor}, api_endpoint={(api_endpoint or 'None')}")
        
        print(f"DEBUG: [get_embedding_function] Using configuration - Model: {model}, Vendor: {vendor}")
        
        # Use the helper function to get the actual embedding function
        return get_embedding_function_by_params(vendor, model, api_key, api_endpoint)
        
    except Exception as e:
        print(f"DEBUG: [get_embedding_function] ERROR: {str(e)}")
        import traceback
        print(f"DEBUG: [get_embedding_function] Stack trace:\n{traceback.format_exc()}")
        raise
    finally:
        db.close()


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
