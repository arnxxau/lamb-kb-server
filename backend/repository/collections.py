"""
Collections repository module for managing collections.

This module provides repository functions for managing collections in both SQLite and ChromaDB.
"""

import os
import json
import logging
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc

from .models import Collection, Visibility, FileRegistry, FileStatus
from .connection import (
    get_db,
    get_chroma_client,
    get_embedding_function,
    get_embedding_function_by_params,
    SessionLocal,
    init_databases,
)

from exceptions import (
    DataIntegrityException,
    DatabaseException,
    ExternalServiceException,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollectionRepository:
    """Repository for managing collections in both SQLite and ChromaDB."""
    
    @staticmethod
    def get_database_status():
        """Get the status of all databases.
        
        Returns:
            A dictionary with database status information
            
        Raises:
            Exception: Any error that occurs during database operations
        """
        db_status = init_databases()
        db = SessionLocal()

        collections_count = db.query(Collection).count()
        
        chroma_client = get_chroma_client()
        chroma_collections = chroma_client.list_collections()
        db.close()

        return {
            "sqlite_status": {
                "initialized": db_status["sqlite_initialized"],
                "schema_valid": db_status["sqlite_schema_valid"],
                "errors": db_status.get("errors", [])
            },
            "chromadb_status": {
                "initialized": db_status["chromadb_initialized"],
                "collections_count": len(chroma_collections)
            },
            "collections_count": collections_count
        }
        

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
        query = db.query(FileRegistry).filter(FileRegistry.collection_id == collection_id)

        if status:
            file_status = FileStatus(status)
            query = query.filter(FileRegistry.status == file_status)

        files = query.all()

        db.close()
        
        return [file.to_dict() for file in files]
            
            
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
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if file_registry:
            file_registry.status = status
            file_registry.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(file_registry)

            db.close()
            return file_registry.to_dict()
        db.close()
        return None
            
    
    @staticmethod
    def create_collection(
        name: str,
        owner: str,
        description: Optional[str],
        visibility: Visibility,
        embeddings_model: Dict[str, Any],
        embedding_function: Callable[[Any], Any]
    ) -> Dict[str, Any]:
        """
        Persist a new collection in both SQLite and ChromaDB using explicit parameters,
        and return a plain dict representation.

        Args:
            name: Collection name
            owner: Owner identifier
            description: Optional text description
            visibility: Visibility enum value
            embeddings_model: Raw embeddings model configuration dict
            embedding_function: Pre-resolved embedding function callable

        Raises:
            ResourceAlreadyExistsException: If a collection with the same name exists

        Returns:
            Dict representing the created collection
        """
        db: Session = SessionLocal()
        try:
            # Enforce unique name
            if db.query(Collection).filter_by(name=name).first():
                raise ResourceAlreadyExistsException(
                    f"Collection with name '{name}' already exists"
                )

            # Prepare ChromaDB collection parameters
            params: Dict[str, Any] = {"name": name, "metadata": {"hnsw:space": "cosine"}}
            params["embedding_function"] = embedding_function

            # Create in ChromaDB
            client = get_chroma_client()
            chroma_collection = client.create_collection(**params)

            # Persist in SQL
            db_collection = Collection(
                name=name,
                owner=owner,
                description=description,
                visibility=visibility,
                embeddings_model=embeddings_model,
                chromadb_uuid=str(chroma_collection.id)
            )
            db.add(db_collection)
            db.commit()
            db.refresh(db_collection)

            # Return plain dict
            return db_collection.to_dict()
        finally:
            db.close()


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
            query = db.query(Collection)
            
            if owner:
                query = query.filter(Collection.owner == owner)
            if visibility:
                query = query.filter(Collection.visibility == visibility)
            
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
        embeddings_model: Optional[Dict[str, Any]] = None,
        rename_from: Optional[str] = None,
        rename_to: Optional[str] = None
    ):
        """
        Apply provided field updates to the Collection record and optionally rename the ChromaDB collection.

        Args:
            collection_id: ID of the collection to update
            name: New collection name
            description: New description
            visibility: New visibility setting
            embeddings_model: Full embeddings_model dict to set
            rename_from: Old Chroma collection name (if rename needed)
            rename_to: New Chroma collection name

        Returns:
            The updated Collection instance, or None if not found
        """
        db: Session = SessionLocal()
        try:
            db_collection = db.query(Collection).get(collection_id)
            if not db_collection:
                return None

            # Apply all non-None field updates
            if name is not None:
                db_collection.name = name
            if description is not None:
                db_collection.description = description
            if visibility is not None:
                db_collection.visibility = visibility
            if embeddings_model is not None:
                db_collection.embeddings_model = embeddings_model

            db.commit()
            db.refresh(db_collection)

            # Handle ChromaDB rename if requested
            if rename_from and rename_to:
                client = get_chroma_client()
                chroma_col = client.get_collection(rename_from)
                chroma_col.modify(name=rename_to)

            return db_collection.to_dict()

        except Exception:
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
def get_file_content(file_id: int) -> str:
    """
    Reconstruct the original file from its ChromaDB chunks.

    Args:
        file_id: primary-key ID in FileRegistry.

    Returns:
        The complete file as a single UTF-8 string.

    Raises:
        ValueError if the file or its chunks cannot be found.
    """
    # --- 1. SQL look-ups ----------------------------------------------------
    with SessionLocal() as db:
        registry = db.get(FileRegistry, file_id)
        if not registry:
            raise ValueError(f"file {file_id} not found")

        collection_row = db.get(Collection, registry.collection_id)
        if not collection_row:
            raise ValueError(f"collection {registry.collection_id} not found")

    # --- 2. Fetch chunks in one call ---------------------------------------
    chroma = get_chroma_client()
    col    = chroma.get_collection(collection_row.name)

    res = col.get(
        where   = {"file_id": file_id},       # indexed filter
        include = ["documents", "metadatas"]  # no embeddings
    )
    if not res["documents"]:
        raise ValueError(f"no chunks stored for file {file_id}")

    # --- 3. Reassemble ------------------------------------------------------
    chunks = sorted(
        zip(res["documents"], res["metadatas"]),
        key=lambda p: p[1].get("chunk_index", 0)
    )
    return "\n".join(text for text, _ in chunks)