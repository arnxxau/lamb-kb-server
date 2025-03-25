"""
Database connection module for SQLite and ChromaDB.

This module provides connection functions for both SQLite (via SQLAlchemy) and ChromaDB.
It handles initialization, connection management, and basic sanity checks.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, List

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import requests

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


def get_embedding_function_by_params(vendor: str, model_name: str, api_key: str = "", api_endpoint: str = ""):
    """Get an embedding function based on vendor and model parameters.
    
    This function returns a ChromaDB-compatible embedding function based on
    the specified vendor and model name.
    
    Args:
        vendor: Embedding vendor (openai, sentence-transformers, ollama)
        model_name: Name of the embedding model
        api_key: API key for the vendor (if needed)
        api_endpoint: Custom API endpoint for the vendor (if needed)
        
    Returns:
        An embedding function compatible with ChromaDB
        
    Raises:
        ValueError: If the vendor is not supported
        RuntimeError: If embeddings service is not available
    """
    import os
    
    print(f"DEBUG: [get_embedding_function_by_params] Getting embedding function for vendor: {vendor}, model: {model_name}")
    
    # Check for empty or None values and use defaults if needed
    if not vendor or vendor == "default":
        vendor = os.getenv("EMBEDDINGS_VENDOR", "sentence_transformers")
        print(f"DEBUG: [get_embedding_function_by_params] Using default vendor from environment: {vendor}")
    
    if not model_name or model_name == "default":
        model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
        print(f"DEBUG: [get_embedding_function_by_params] Using default model from environment: {model_name}")
        
    # Convert vendor to lowercase for case-insensitive matching
    vendor = vendor.lower()
    
    # Ollama embedding function
    if vendor in ("ollama", "local"):
        import requests
        
        if not api_endpoint or api_endpoint == "default":
            api_endpoint = os.getenv("OLLAMA_API_ENDPOINT", "http://localhost:11434")
            print(f"DEBUG: [get_embedding_function_by_params] Using Ollama embeddings with model: {model_name}")
            print(f"DEBUG: [get_embedding_function_by_params] Using endpoint from environment: {api_endpoint}")
        
        # Test the Ollama API before proceeding
        try:
            print(f"DEBUG: [get_embedding_function_by_params] Testing Ollama embeddings API at: {api_endpoint}")
            response = requests.get(f"{api_endpoint}/api/embeddings")
            if response.status_code == 200:
                print(f"DEBUG: [get_embedding_function_by_params] Ollama embeddings API is working properly")
            else:
                print(f"WARNING: [get_embedding_function_by_params] Ollama API returned status code: {response.status_code}")
        except Exception as e:
            print(f"WARNING: [get_embedding_function_by_params] Failed to connect to Ollama service: {str(e)}")
            print(f"WARNING: [get_embedding_function_by_params] Will try to use anyway assuming it might be available for query/embedding")
        
        # Use a class to implement the __call__ interface that ChromaDB expects
        class OllamaEmbeddingFunction:
            def __init__(self, model_name, api_endpoint):
                self.model_name = model_name
                self.api_endpoint = api_endpoint
                
            def __call__(self, input):
                """Generate embeddings using Ollama with the expected ChromaDB interface."""
                if isinstance(input, str):
                    input = [input]
                    
                embeddings = []
                for text in input:
                    try:
                        response = requests.post(
                            f"{self.api_endpoint}/api/embeddings",
                            json={"model": self.model_name, "prompt": text}
                        )
                        
                        if response.status_code == 200:
                            embedding = response.json().get("embedding", [])
                            # Print dimensionality info for debugging
                            print(f"DEBUG: [ollama_embeddings] Generated embedding with {len(embedding)} dimensions")
                            embeddings.append(embedding)
                        else:
                            raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to generate Ollama embedding: {str(e)}")
                
                return embeddings
        
        return OllamaEmbeddingFunction(model_name, api_endpoint)
    
    # Sentence Transformers embedding function (local)
    elif vendor in ("sentence_transformers", "sentence-transformers", "st", "hf", "huggingface"):
        print(f"DEBUG: [get_embedding_function_by_params] Using local sentence-transformers with model: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Create a class with proper __call__ interface
            class SentenceTransformerEmbeddingFunction:
                def __init__(self, model_name):
                    self.model = SentenceTransformer(model_name)
                    
                def __call__(self, input):
                    """Generate embeddings using sentence-transformers with the expected ChromaDB interface."""
                    if isinstance(input, str):
                        input = [input]
                    
                    embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
                    # Print dimensionality info for debugging
                    print(f"DEBUG: [sentence_transformer_embeddings] Generated {len(embeddings)} embeddings with {len(embeddings[0]) if embeddings else 0} dimensions")
                    return embeddings
            
            return SentenceTransformerEmbeddingFunction(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize sentence-transformers: {str(e)}")
    
    # OpenAI embedding function
    elif vendor == "openai":
        print(f"DEBUG: [get_embedding_function_by_params] Using OpenAI embeddings with model: {model_name}")
        
        try:
            import openai
            
            # Set up API key
            if api_key and api_key != "default":
                openai_api_key = api_key
            else:
                # Check both environment variables
                openai_api_key = os.getenv("OPENAI_API_KEY", "")
                if not openai_api_key:
                    # Fall back to EMBEDDINGS_APIKEY if OPENAI_API_KEY is not set
                    openai_api_key = os.getenv("EMBEDDINGS_APIKEY", "")
                    if openai_api_key:
                        print(f"DEBUG: [get_embedding_function_by_params] Using EMBEDDINGS_APIKEY since OPENAI_API_KEY is not set")
                
            if not openai_api_key:
                raise ValueError("OpenAI API key is required but not provided in either OPENAI_API_KEY or EMBEDDINGS_APIKEY")
                
            # Set up custom API endpoint if provided
            if api_endpoint and api_endpoint != "default":
                openai.api_base = api_endpoint
            
            # Create a class with proper __call__ interface
            class OpenAIEmbeddingFunction:
                def __init__(self, model_name, api_key):
                    self.model_name = model_name
                    # No logging of API key details at all
                    print(f"DEBUG: [OpenAIEmbeddingFunction.__init__] Creating OpenAI client")
                    
                    # Create client with the API key - pass it directly without manipulation
                    self.client = openai.OpenAI(api_key=api_key)
                    
                def __call__(self, input):
                    """Generate embeddings using OpenAI with the expected ChromaDB interface."""
                    if isinstance(input, str):
                        input = [input]
                        
                    all_embeddings = []
                    
                    # Process in batches to avoid rate limits
                    batch_size = 10
                    for i in range(0, len(input), batch_size):
                        batch = input[i:i+batch_size]
                        try:
                            response = self.client.embeddings.create(
                                model=self.model_name,
                                input=batch
                            )
                            
                            batch_embeddings = [embedding.embedding for embedding in response.data]
                            all_embeddings.extend(batch_embeddings)
                        except Exception as e:
                            print(f"ERROR: [OpenAIEmbeddingFunction.__call__] Failed to generate embeddings: {str(e)}")
                            raise RuntimeError(f"OpenAI API error: {str(e)}")
                    
                    # Print dimensionality info for debugging
                    print(f"DEBUG: [openai_embeddings] Generated {len(all_embeddings)} embeddings with {len(all_embeddings[0]) if all_embeddings else 0} dimensions")
                    return all_embeddings
            
            return OpenAIEmbeddingFunction(model_name, openai_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI embeddings: {str(e)}")
    
    else:
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


def get_embedding_function(collection_id_or_obj: Union[int, Collection, Dict[str, Any]]) -> Callable:
    """Get the embedding function for a collection by its ID or Collection object.
    
    This function fetches the embedding configuration from the database
    and constructs the appropriate embedding function.
    
    Args:
        collection_id_or_obj: Either a collection ID, a Collection object, or a dict from CollectionService.get_collection
        
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
    
    print(f"DEBUG: [get_embedding_function] Getting embedding function for collection: {collection_id_or_obj}")
    
    # Get SQLite database session
    db = next(get_db())
    
    try:
        # Determine if we got an ID or a Collection object
        collection = None
        collection_id = None
        
        if isinstance(collection_id_or_obj, Collection):
            # We already have the collection object
            collection = collection_id_or_obj
            collection_id = collection.id
            print(f"DEBUG: [get_embedding_function] Using provided Collection object (id={collection.id})")
        elif isinstance(collection_id_or_obj, int):
            # Get the collection from the database
            collection_id = collection_id_or_obj
            collection = db.query(Collection).filter(Collection.id == collection_id).first()
            if not collection:
                print(f"DEBUG: [get_embedding_function] ERROR: Collection not found with ID: {collection_id}")
                raise ValueError(f"Collection not found with ID: {collection_id}")
            print(f"DEBUG: [get_embedding_function] Retrieved Collection from database (id={collection.id})")
        elif isinstance(collection_id_or_obj, dict):
            # Handle dict-like objects (e.g. from CollectionService.get_collection)
            collection_id = collection_id_or_obj.get('id')
            if not collection_id:
                raise ValueError("Collection dictionary must contain an 'id' field")
            
            # Get the embedding model directly from the dict if available
            if 'embeddings_model' in collection_id_or_obj:
                embedding_config = collection_id_or_obj['embeddings_model']
                
                # Get configuration parameters
                model = embedding_config.get("model")
                vendor = embedding_config.get("vendor")
                api_key = embedding_config.get("apikey")
                api_endpoint = embedding_config.get("api_endpoint")
                
                print(f"DEBUG: [get_embedding_function] Using embedding config from dict - Model: {model}, Vendor: {vendor}")
                
                # Use the helper function to get the actual embedding function
                return get_embedding_function_by_params(vendor, model, api_key, api_endpoint)
            
            # Otherwise get the collection from the database
            collection = db.query(Collection).filter(Collection.id == collection_id).first()
            if not collection:
                print(f"DEBUG: [get_embedding_function] ERROR: Collection not found with ID: {collection_id}")
                raise ValueError(f"Collection not found with ID: {collection_id}")
            print(f"DEBUG: [get_embedding_function] Retrieved Collection from database (id={collection.id})")
        else:
            raise ValueError(f"Expected Collection object, dictionary or ID, got {type(collection_id_or_obj)}")
            
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
            
            # Attempt to resolve any remaining 'default' values from environment
            if model == "default":
                model = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                print(f"WARNING: [get_embedding_function] Resolved default model to: {model}")
                
            if vendor == "default":
                vendor = os.getenv("EMBEDDINGS_VENDOR", "sentence_transformers")
                print(f"WARNING: [get_embedding_function] Resolved default vendor to: {vendor}")
                
            if api_key == "default":
                api_key = os.getenv("EMBEDDINGS_APIKEY", "")
                print(f"WARNING: [get_embedding_function] Resolved default API key from environment")
                
            if api_endpoint == "default":
                api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT", None)
                print(f"WARNING: [get_embedding_function] Resolved default API endpoint to: {api_endpoint}")
        
        print(f"DEBUG: [get_embedding_function] Using configuration - Model: {model}, Vendor: {vendor}")
        
        # Use the helper function to get the actual embedding function
        embedding_function = get_embedding_function_by_params(vendor, model, api_key, api_endpoint)
        
        # Test the embedding function to ensure it works
        try:
            test_result = embedding_function(["Test embedding function validation"])
            print(f"DEBUG: [get_embedding_function] Test successful with {len(test_result[0])} dimensions")
        except Exception as test_error:
            print(f"ERROR: [get_embedding_function] Embedding function test failed: {str(test_error)}")
            # We re-raise the error since a non-working embedding function is useless
            raise
        
        return embedding_function
        
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
