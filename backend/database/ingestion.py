"""
Database repository module for managing file ingestion.

This module provides repository functions for managing file ingestion operations in both SQLite and ChromaDB.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from sqlalchemy.orm import Session

from .models import Collection, FileRegistry, FileStatus
from .connection import get_db, get_chroma_client, get_embedding_function
from .service import CollectionRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionRepository:
    """Repository for managing file ingestion in both SQLite and ChromaDB."""

    @staticmethod
    def register_file(
        db: Session, 
        collection_id: int, 
        file_path: str, 
        file_url: str, 
        original_filename: str, 
        plugin_name: str, 
        plugin_params: Dict[str, Any],
        owner: str,
        document_count: int = 0,
        content_type: Optional[str] = None,
        status: FileStatus = FileStatus.COMPLETED
    ) -> FileRegistry:
        """Register a file in the FileRegistry table.
        
        Args:
            db: Database session
            collection_id: ID of the collection
            file_path: Path to the file on the server
            file_url: URL to access the file
            original_filename: Original name of the file
            plugin_name: Name of the plugin used for ingestion
            plugin_params: Parameters used for ingestion
            owner: Owner of the file
            document_count: Number of chunks/documents created
            content_type: MIME type of the file
            status: Status of the file (default: COMPLETED)
            
        Returns:
            The created FileRegistry entry
        """
        # Get file size
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Ensure plugin_params is a dict, not a string
        if isinstance(plugin_params, str):
            try:
                plugin_params = json.loads(plugin_params)
            except:
                plugin_params = {}
                
        # Create the file registry entry
        file_registry = FileRegistry(
            collection_id=collection_id,
            original_filename=original_filename,
            file_path=file_path,
            file_url=file_url,
            file_size=file_size,
            content_type=content_type,
            plugin_name=plugin_name,
            plugin_params=plugin_params,
            status=status,
            document_count=document_count,
            owner=owner
        )
        
        db.add(file_registry)
        db.commit()
        db.refresh(file_registry)
        
        return file_registry
    
    @staticmethod
    def update_file_status(db: Session, file_id: int, status: FileStatus) -> Optional[FileRegistry]:
        """Update the status of a file in the registry.
        
        Args:
            db: Database session
            file_id: ID of the file registry entry
            status: New status
            
        Returns:
            The updated FileRegistry entry or None if not found
        """
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if file_registry:
            file_registry.status = status
            file_registry.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(file_registry)
        
        return file_registry
    
    @staticmethod
    def add_documents_to_collection(
        db: Session,
        collection_id: int,
        documents: List[Dict[str, Any]],
        embeddings_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Add documents to a ChromaDB collection.
        
        Args:
            db: Database session
            collection_id: ID of the collection
            documents: List of document chunks with metadata
            embeddings_function: Optional custom embeddings function
            
        Returns:
            Status information about the ingestion operation
            
        Raises:
            ValueError: If collection not found or adding documents fails
        """
        print(f"DEBUG: [add_documents_to_collection] Starting for collection_id: {collection_id}")
        print(f"DEBUG: [add_documents_to_collection] Number of documents: {len(documents)}")
        
        # Get the collection
        print(f"DEBUG: [add_documents_to_collection] Getting collection from database")
        db_collection = CollectionRepository.get_collection(collection_id)
        if not db_collection:
            print(f"DEBUG: [add_documents_to_collection] ERROR: Collection not found")
            raise ValueError(f"Collection with ID {collection_id} not found")
        
        # Get collection name and attributes - handle both dict-like and attribute access
        collection_name = db_collection['name'] if isinstance(db_collection, dict) else db_collection.name
        
        print(f"DEBUG: [add_documents_to_collection] Found collection: {collection_name}")
        
        # Store the embedding config for logging/debugging
        embedding_config = db_collection.embeddings_model if not isinstance(db_collection, dict) else db_collection['embeddings_model']
        print(f"DEBUG: [add_documents_to_collection] Embedding config: {embedding_config}")
        
        # Extract key embedding model parameters for verification
        vendor = embedding_config.get("vendor", "")
        model_name = embedding_config.get("model", "")
        print(f"DEBUG: [add_documents_to_collection] Using embeddings - vendor: {vendor}, model: {model_name}")
        
        # Get ChromaDB client
        print(f"DEBUG: [add_documents_to_collection] Getting ChromaDB client")
        chroma_client = get_chroma_client()
        
        # Get the embedding function for this collection using the collection
        collection_embedding_function = None
        try:
            print(f"DEBUG: [add_documents_to_collection] Creating embedding function from collection record")
            # Pass collection or collection_id to get_embedding_function
            collection_embedding_function = get_embedding_function(db_collection)
            print(f"DEBUG: [add_documents_to_collection] Created embedding function: {collection_embedding_function is not None}")
            
            # Test the embedding function to verify it works
            test_result = collection_embedding_function(["Test embedding function verification"])
            print(f"DEBUG: [add_documents_to_collection] Embedding function test successful, dimensions: {len(test_result[0])}")
        except Exception as ef_e:
            print(f"DEBUG: [add_documents_to_collection] ERROR creating embedding function: {str(ef_e)}")
            import traceback
            print(f"DEBUG: [add_documents_to_collection] Stack trace:\n{traceback.format_exc()}")
            raise ValueError(f"Failed to create embedding function: {str(ef_e)}")
        
        # Verify ChromaDB collection exists (do not recreate it)
        try:
            print(f"DEBUG: [add_documents_to_collection] Verifying ChromaDB collection exists: {collection_name}")
            
            # First check if collection exists in ChromaDB
            collections = chroma_client.list_collections()
            
            # In ChromaDB v0.6.0+, list_collections returns a list of collection names (strings)
            # In older versions, it returned objects with a name attribute
            if collections and isinstance(collections[0], str):
                # ChromaDB v0.6.0+ - collections is a list of strings
                collection_exists = collection_name in collections
                print(f"DEBUG: [add_documents_to_collection] Using ChromaDB v0.6.0+ API: collections are strings")
            else:
                # Older ChromaDB - collections is a list of objects with name attribute
                try:
                    collection_exists = any(col.name == collection_name for col in collections)
                    print(f"DEBUG: [add_documents_to_collection] Using older ChromaDB API: collections have name attribute")
                except (AttributeError, NotImplementedError):
                    # Fall back to checking if we can get the collection
                    try:
                        chroma_client.get_collection(name=collection_name)
                        collection_exists = True
                        print(f"DEBUG: [add_documents_to_collection] Verified collection exists by get_collection")
                    except Exception:
                        collection_exists = False
            
            if not collection_exists:
                print(f"ERROR: [add_documents_to_collection] ChromaDB collection does not exist: {collection_name}")
                raise ValueError(
                    f"Collection '{collection_name}' exists in database but not in ChromaDB. "
                    f"This indicates data inconsistency. Please recreate the collection."
                )
            
            # Get the ChromaDB collection with our embedding function to ensure consistent embeddings
            chroma_collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=collection_embedding_function
            )
            print(f"DEBUG: [add_documents_to_collection] ChromaDB collection retrieved successfully")
            
            # Verify embedding model configuration consistency
            existing_metadata = chroma_collection.metadata
            if existing_metadata:
                # Extract embedding model info from metadata if available
                existing_embeddings_model = existing_metadata.get("embeddings_model", "{}")
                try:
                    if isinstance(existing_embeddings_model, str):
                        existing_config = json.loads(existing_embeddings_model)
                    else:
                        existing_config = existing_embeddings_model
                        
                    existing_vendor = existing_config.get("vendor", "")
                    existing_model = existing_config.get("model", "")
                    
                    # Compare with what we expect
                    if existing_vendor != vendor or existing_model != model_name:
                        print(f"WARNING: [add_documents_to_collection] Embedding model mismatch detected!")
                        print(f"WARNING: ChromaDB collection uses - vendor: {existing_vendor}, model: {existing_model}")
                        print(f"WARNING: SQLite record specifies - vendor: {vendor}, model: {model_name}")
                        print(f"WARNING: Will use the embedding function from SQLite record")
                        # We continue with the SQLite embedding function but log the warning
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"WARNING: [add_documents_to_collection] Could not parse embedding model from metadata: {str(e)}")
        except ValueError:
            # Pass through our specific exceptions
            raise
        except Exception as e:
            print(f"DEBUG: [add_documents_to_collection] ERROR with ChromaDB collection: {str(e)}")
            import traceback
            print(f"DEBUG: [add_documents_to_collection] Stack trace:\n{traceback.format_exc()}")
            raise ValueError(f"Failed to access ChromaDB collection: {str(e)}")
        
        # Prepare documents for ChromaDB
        print(f"DEBUG: [add_documents_to_collection] Preparing documents for ChromaDB")
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = f"{uuid.uuid4().hex}"
            ids.append(doc_id)
            texts.append(doc["text"])
            
            # Add document ID and timestamp to metadata
            metadata = doc.get("metadata", {}).copy()
            metadata["document_id"] = doc_id
            metadata["ingestion_timestamp"] = datetime.utcnow().isoformat()
            
            # Add embedding model information to each document's metadata for debugging
            metadata["embedding_vendor"] = vendor
            metadata["embedding_model"] = model_name
            
            metadatas.append(metadata)
        
        print(f"DEBUG: [add_documents_to_collection] Prepared {len(ids)} documents")
        
        # Add documents to ChromaDB collection
        try:
            print(f"DEBUG: [add_documents_to_collection] Adding documents to ChromaDB")
            if len(texts) > 0:
                print(f"DEBUG: [add_documents_to_collection] First document sample: {texts[0][:100]}...")
            
            # Add timing for performance monitoring
            import time
            start_time = time.time()
            
            # Add documents in smaller batches for better error handling and progress tracking
            batch_size = 5
            
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                
                print(f"DEBUG: [add_documents_to_collection] Processing batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
                
                batch_start_time = time.time()
                
                # Add documents - ALWAYS use the collection embedding function from SQLite record for consistency
                chroma_collection.add(
                    ids=ids[i:batch_end],
                    documents=texts[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )
                
                batch_end_time = time.time()
                print(f"DEBUG: [add_documents_to_collection] Batch {i//batch_size + 1} completed in {batch_end_time - batch_start_time:.2f} seconds")
            
            end_time = time.time()
            print(f"DEBUG: [add_documents_to_collection] ChromaDB add operation completed in {end_time - start_time:.2f} seconds")
            
            return {
                "collection_id": collection_id,
                "collection_name": collection_name,
                "documents_added": len(documents),
                "success": True,
                "embedding_info": {
                    "vendor": vendor,
                    "model": model_name
                }
            }
        except Exception as e:
            print(f"DEBUG: [add_documents_to_collection] ERROR adding documents to ChromaDB: {str(e)}")
            import traceback
            print(f"DEBUG: [add_documents_to_collection] Stack trace:\n{traceback.format_exc()}")
            
            # Try to provide more specific error information
            error_message = str(e)
            if "api_key" in error_message.lower() or "apikey" in error_message.lower():
                print(f"DEBUG: [add_documents_to_collection] Likely an API key issue with embeddings")
                raise ValueError("Failed to add documents: API key issue with embeddings provider. Check your API key configuration.")
            elif "timeout" in error_message.lower():
                print(f"DEBUG: [add_documents_to_collection] Request timeout detected")
                raise ValueError("Failed to add documents: Timeout when generating embeddings. The embedding service may be unavailable.")
            else:
                raise ValueError(f"Failed to add documents to collection: {str(e)}")