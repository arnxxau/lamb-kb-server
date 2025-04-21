"""
Database repository module for managing collections.

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
            print(f"ERROR: [create_collection] {error_msg}")
            raise ValueError(error_msg)
        
        # Create ChromaDB collection
        chroma_client = get_chroma_client()
        
        # Get the appropriate embedding function based on model configuration
        embedding_func = None
        if embeddings_model:
            print(f"DEBUG: [create_collection] Using embedding config from dict - Model: {embeddings_model['model']}, Vendor: {embeddings_model['vendor']}")
            try:
                embedding_func = get_embedding_function_by_params(
                    vendor=embeddings_model['vendor'],
                    model_name=embeddings_model['model'],
                    api_key=embeddings_model.get('apikey', ''),
                    api_endpoint=embeddings_model.get('api_endpoint', '')
                )
            except Exception as e:
                print(f"ERROR: [create_collection] Failed to create embedding function: {str(e)}")
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
            print(f"DEBUG: [create_collection] Successfully created ChromaDB collection")
            
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
            print(f"ERROR: [create_collection] Failed to create ChromaDB collection: {str(e)}")
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
        apikey: Optional[str] = None
    ) -> Optional[Collection]:
        """
        Update a collection's SQLite record and rename ChromaDB collection if needed.

        Args:
            collection_id: ID of the collection to update
            name: New collection name
            description: New description
            visibility: New visibility setting
            model: (NOT SUPPORTED) Attempt to change embeddings model name
            vendor: (NOT SUPPORTED) Attempt to change embeddings vendor
            endpoint: New embeddings-model endpoint
            apikey: New embeddings-model API key
        """
        db = SessionLocal()
        try:
            db_collection = db.query(Collection).get(collection_id)
            if not db_collection:
                return None

            old_name = db_collection.name

            # Disallow changing model or vendor
            current_conf = db_collection.embeddings_model or {}
            if model is not None and model != current_conf.get('model'):
                logger.warning("Changing 'model' is not supported and will be ignored.")
            if vendor is not None and vendor != current_conf.get('vendor'):
                logger.warning("Changing 'vendor' is not supported and will be ignored.")

            if name is not None:
                db_collection.name = name
            if description is not None:
                db_collection.description = description
            if visibility is not None:
                db_collection.visibility = visibility

            # Only update endpoint and apikey
            if endpoint is not None:
                current_conf['api_endpoint'] = endpoint
            if apikey is not None:
                current_conf['apikey'] = apikey
            db_collection.embeddings_model = current_conf

            db.commit()
            db.refresh(db_collection)

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
                print(f"Error deleting ChromaDB collection: {e}")
            
            # Delete from SQLite
            db.delete(db_collection)
            db.commit()
            
            return True
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()