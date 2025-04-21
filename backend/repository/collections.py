"""
Collections repository module for managing collections.

This module provides repository functions for managing collections in both SQLite and ChromaDB.
"""

import os
import json
import logging
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc

from .models import Collection, Visibility, FileRegistry, FileStatus
from .connection import get_db, get_chroma_client, get_embedding_function, get_embedding_function_by_params, SessionLocal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollectionRepository:
    """Repository for managing collections in both SQLite and ChromaDB."""

    @staticmethod
    def list_files(collection_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files in a collection with optional status filtering.
        
        Args:
            collection_id: ID of the collection
            status: Optional file status to filter by
            
        Returns:
            List of file registry entries
        """
        db = SessionLocal()
        try:
            # Query files
            query = db.query(FileRegistry).filter(FileRegistry.collection_id == collection_id)
            
            # Apply status filter if provided
            if status:
                try:
                    file_status = FileStatus(status)
                    query = query.filter(FileRegistry.status == file_status)
                except ValueError:
                    # Handle validation at the service level
                    pass
            
            # Get results
            files = query.all()
            
            return [file.to_dict() for file in files]
        finally:
            db.close()
            
    @staticmethod
    def update_file_status(file_id: int, status: FileStatus) -> Optional[Dict[str, Any]]:
        """Update the status of a file in the registry.
        
        Args:
            file_id: ID of the file registry entry
            status: New status
            
        Returns:
            Updated file registry entry or None if not found
        """
        db = SessionLocal()
        try:
            file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
            if file_registry:
                file_registry.status = status
                file_registry.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(file_registry)
                return file_registry.to_dict()
            return None
        finally:
            db.close()
    
    @staticmethod
    def create_collection(
        name: str,
        owner: str,
        description: Optional[str] = None,
        visibility: Visibility = Visibility.PRIVATE,
        embeddings_model: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Create a new collection in both SQLite and ChromaDB.
        
        Args:
            db: SQLAlchemy database session
            name: Name of the collection
            owner: Owner of the collection
            description: Optional description
            visibility: Visibility setting (private or public)
            embeddings_model: Optional custom embeddings model configuration
            
        Returns:
            The created Collection object
        """
        # embeddings_model=None should never happen from the API
        if embeddings_model is None:
            error_msg = "No embeddings model configuration provided. This is required for collection creation."
            logger.error(f"[create_collection] {error_msg}")
            raise ValueError(error_msg)
        
        # Create ChromaDB collection
        chroma_client = get_chroma_client()
        
        # Get the appropriate embedding function based on model configuration
        embedding_func = None
        if embeddings_model:
            logger.debug(f"Using embedding config from dict - Model: {embeddings_model['model']}, Vendor: {embeddings_model['vendor']}")
            try:
                embedding_func = get_embedding_function_by_params(
                    vendor=embeddings_model['vendor'],
                    model_name=embeddings_model['model'],
                    api_key=embeddings_model.get('apikey', ''),
                    api_endpoint=embeddings_model.get('api_endpoint', '')
                )
            except Exception as e:
                logger.error(f"Failed to create embedding function: {str(e)}")
                raise ValueError(f"Failed to create embedding function: {str(e)}")
        
        # Prepare collection parameters
        collection_params = {
            "name": name,
            "metadata": {
                "hnsw:space": "cosine",
            }
        }
        
        if embedding_func:
            collection_params["embedding_function"] = embedding_func
        
        try:
            chroma_collection = chroma_client.create_collection(**collection_params)
            logger.debug(f"Successfully created ChromaDB collection")
            
            # Create SQLite record with a new DB session
            db = SessionLocal()
            try:
                db_collection = Collection(
                    name=name,
                    description=description,
                    owner=owner,
                    visibility=visibility,
                    embeddings_model=embeddings_model,
                    chromadb_uuid=str(chroma_collection.id)
                )
                db.add(db_collection)
                db.commit()
                db.refresh(db_collection)
                return db_collection
            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {str(e)}")
            raise
    

    @staticmethod
    def get_collection(collection_id: int) -> Optional[Dict[str, Any]]:
        """Get a collection by ID.
        
        Args:
            collection_id: ID of the collection to retrieve
            
        Returns:
            The Collection as a dictionary if found, None otherwise
        """
        db = SessionLocal()
        try:
            collection = db.query(Collection).filter(Collection.id == collection_id).first()
            if not collection:
                return None
            
            collection_dict = collection.to_dict()
            return collection_dict
        finally:
            db.close()
    
    @staticmethod
    def get_collection_by_name(name: str) -> Optional[Collection]:
        """Get a collection by name.
        
        Args:
            name: Name of the collection to retrieve
            
        Returns:
            The Collection object if found, None otherwise
        """
        db = SessionLocal()
        try:
            return db.query(Collection).filter(Collection.name == name).first()
        finally:
            db.close()
    
    @staticmethod
    def get_collection_by_chromadb_uuid(chromadb_uuid: str) -> Optional[Collection]:
        """Get a collection by ChromaDB UUID.
        
        Args:
            chromadb_uuid: ChromaDB UUID of the collection to retrieve
            
        Returns:
            The Collection object if found, None otherwise
        """
        db = SessionLocal()
        try:
            return db.query(Collection).filter(Collection.chromadb_uuid == chromadb_uuid).first()
        finally:
            db.close()
    
    @staticmethod
    def list_collections(
        owner: Optional[str] = None,
        visibility: Optional[Visibility] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List collections with optional filtering.
        
        Args:
            owner: Optional filter by owner
            visibility: Optional filter by visibility
            skip: Number of collections to skip
            limit: Maximum number of collections to return
            
        Returns:
            List of collections
        """
        db = SessionLocal()
        try:
            # Build query
            query = db.query(Collection)
            
            # Apply filters if provided
            if owner:
                query = query.filter(Collection.owner == owner)
            if visibility:
                query = query.filter(Collection.visibility == visibility)
            
            # Apply pagination
            collections = query.offset(skip).limit(limit).all()

            return [col.to_dict() for col in collections]
        finally:
            db.close()
    
    @staticmethod
    def update_collection(
        collection_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[Visibility] = None,
        model: Optional[str] = None,
        vendor: Optional[str] = None,
        endpoint: Optional[str] = None,
        apikey: Optional[str] = None,
        embeddings_model_updates: Optional[Dict[str, Any]] = None
    ) -> Optional[Collection]:
        """
        Update a collection's SQLite record and rename ChromaDB collection if needed.

        Args:
            collection_id: ID of the collection to update
            name: New collection name
            description: New description
            visibility: New visibility setting
            model: New embeddings model name (only for backward compatibility)
            vendor: New embeddings vendor (only for backward compatibility)
            endpoint: New embeddings-model endpoint (only for backward compatibility)
            apikey: New embeddings-model API key (only for backward compatibility)
            embeddings_model_updates: Dict of updates to apply to the embeddings_model field
        
        Returns:
            Updated Collection or None if not found
        """
        db = SessionLocal()
        try:
            db_collection = db.query(Collection).get(collection_id)
            if not db_collection:
                return None

            old_name = db_collection.name

            # Update basic properties
            if name is not None:
                db_collection.name = name
            if description is not None:
                db_collection.description = description
            if visibility is not None:
                db_collection.visibility = visibility

            # Update embeddings model if provided
            if embeddings_model_updates:
                current_conf = db_collection.embeddings_model or {}
                current_conf.update(embeddings_model_updates)
                db_collection.embeddings_model = current_conf
            # For backward compatibility (will be processed by service layer)
            else:
                current_conf = db_collection.embeddings_model or {}
                # Apply endpoint and API key if provided
                if endpoint is not None:
                    current_conf['api_endpoint'] = endpoint
                if apikey is not None:
                    current_conf['apikey'] = apikey
                db_collection.embeddings_model = current_conf

            db.commit()
            db.refresh(db_collection)

            # Update ChromaDB collection name if it changed
            if name and name != old_name:
                client = get_chroma_client()
                chroma_col = client.get_collection(old_name)
                chroma_col.modify(name=name)

            return db_collection
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    
    @staticmethod
    def delete_collection(collection_id: int) -> bool:
        """Delete a collection from both SQLite and ChromaDB.
        
        Args:
            collection_id: ID of the collection to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        db = SessionLocal()
        try:
            db_collection = db.query(Collection).filter(Collection.id == collection_id).first()
            
            if not db_collection:
                return False
            
            # Delete from ChromaDB
            collection_name = db_collection.name
            chroma_client = get_chroma_client()
            try:
                chroma_client.delete_collection(collection_name)
            except Exception as e:
                logger.error(f"Error deleting ChromaDB collection: {e}")
            
            # Delete from SQLite
            db.delete(db_collection)
            db.commit()
            
            return True
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
            
    @staticmethod
    def get_file_content(file_id: int) -> Dict[str, Any]:
        """Get the content of a file from file registry.
        
        Args:
            file_id: ID of the file registry entry
            
        Returns:
            Content of the file with metadata
            
        Raises:
            ValueError: If file or collection not found or content cannot be retrieved
        """
        db = SessionLocal()
        try:
            # Get file registry entry
            file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
            if not file_registry:
                raise ValueError(f"File with ID {file_id} not found")
            
            # Get the file content based on the plugin type
            file_content = ""
            content_type = "text"
            
            # Get collection for any file type
            collection = db.query(Collection).filter(Collection.id == file_registry.collection_id).first()
            if not collection:
                raise ValueError(f"Collection with ID {file_registry.collection_id} not found")
                
            # Get database content from ChromaDB
            # Get ChromaDB client and collection
            chroma_client = get_chroma_client()
            chroma_collection = chroma_client.get_collection(name=collection.name)
            
            # Get the source filename/url
            source = file_registry.original_filename
            
            # Get the file type and plugin name
            plugin_name = file_registry.plugin_name
            
            # Set the appropriate file type based on extension or plugin
            original_extension = ""
            if "." in source:
                original_extension = source.split(".")[-1].lower()
                
            # Determine content type based on file extension or plugin name
            if plugin_name == "url_ingest" or source.startswith(("http://", "https://")):
                content_type = "markdown"  # URL content from firecrawl is markdown
            elif original_extension in ["md", "markdown"]:
                content_type = "markdown"
            elif original_extension in ["txt", "text"]:
                content_type = "text"
            else:
                content_type = "text"  # Default to text
                
            # Query ChromaDB for documents with this source
            results = chroma_collection.get(
                where={"source": source}, 
                include=["documents", "metadatas"]
            )
            
            if not results or not results["documents"] or len(results["documents"]) == 0:
                # Try other fields if "source" doesn't work
                results = chroma_collection.get(
                    where={"filename": source}, 
                    include=["documents", "metadatas"]
                )
                
            if not results or not results["documents"] or len(results["documents"]) == 0:
                # For regular files, the source field might be the file path
                # Try to find by just the filename part
                import os
                filename = os.path.basename(source)
                if filename:
                    results = chroma_collection.get(
                        where={"filename": filename}, 
                        include=["documents", "metadatas"]
                    )
                    
            if not results or not results["documents"] or len(results["documents"]) == 0:
                # If we still haven't found it, check if there's a physical file we can read
                if os.path.exists(file_registry.file_path) and os.path.isfile(file_registry.file_path):
                    try:
                        with open(file_registry.file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            
                        return {
                            "file_id": file_id,
                            "original_filename": source,
                            "content": file_content,
                            "content_type": content_type,
                            "chunk_count": 1,
                            "timestamp": file_registry.updated_at.isoformat() if file_registry.updated_at else None
                        }
                    except Exception as e:
                        logger.error(f"Error reading file from disk: {str(e)}", exc_info=True)
                        # Continue to error handling
                
                raise ValueError(f"No content found for file: {source}")
            
            # Reconstruct the content from chunks
            # First sort by chunk_index
            chunk_docs = []
            for i, doc in enumerate(results["documents"]):
                if i < len(results["metadatas"]) and results["metadatas"][i]:
                    metadata = results["metadatas"][i]
                    chunk_docs.append({
                        "text": doc,
                        "index": metadata.get("chunk_index", i),
                        "count": metadata.get("chunk_count", 0)
                    })
            
            # Sort chunks by index
            chunk_docs.sort(key=lambda x: x["index"])
            
            # Join all chunks
            full_content = "\n".join(doc["text"] for doc in chunk_docs)
            
            # Return content with metadata
            return {
                "file_id": file_id,
                "original_filename": source,
                "content": full_content,
                "content_type": content_type,
                "chunk_count": len(chunk_docs),
                "timestamp": file_registry.updated_at.isoformat() if file_registry.updated_at else None
            }
        except Exception as e:
            logger.error(f"Error retrieving file content: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to retrieve file content: {str(e)}")
        finally:
            db.close()