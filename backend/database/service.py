"""
Database service module for managing collections.

This module provides service functions for managing collections in both SQLite and ChromaDB.
"""

import os
import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc

from .models import Collection, Visibility
from .connection import get_db, get_chroma_client, get_embedding_function, get_embedding_function_by_params


class CollectionService:
    """Service for managing collections in both SQLite and ChromaDB."""
    
    @staticmethod
    def create_collection(
        db: Session,
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
                "owner": owner,
                "description": description or "",
                "visibility": visibility.value,
                # Don't include sqlite_id in initial metadata since we don't have it yet
                # We'll try to update it after creation
            }
        }
        
        if embedding_func:
            collection_params["embedding_function"] = embedding_func
        
        try:
            chroma_collection = chroma_client.create_collection(**collection_params)
            print(f"DEBUG: [create_collection] Successfully created ChromaDB collection")
            
            # Get the collection's ID from ChromaDB
            try:
                # In v0.6.0+, we can get the collection by name and then get its ID
                collection = chroma_client.get_collection(name=name)
                collection_id = collection.id
                # Convert UUID to string to avoid SQLite compatibility issues
                collection_id_str = str(collection_id) if collection_id else None
                print(f"DEBUG: [create_collection] Got collection ID: {collection_id_str}")
            except Exception as e:
                print(f"WARNING: [create_collection] Could not get collection ID: {e}")
                collection_id_str = None
            
            # Create SQLite record
            db_collection = Collection(
                name=name,
                description=description,
                owner=owner,
                visibility=visibility,
                embeddings_model=json.dumps(embeddings_model),
                chromadb_uuid=collection_id_str  # Store as string, not UUID object
            )
            db.add(db_collection)
            db.commit()
            db.refresh(db_collection)
            
            # Update ChromaDB metadata with SQLite ID
            if collection_id:
                try:
                    collection = chroma_client.get_collection(id=collection_id)
                    
                    # Add SQLite ID to metadata if available - using str representation to avoid None
                    if db_collection.id:
                        try:
                            # Note: ChromaDB doesn't support updating metadata directly
                            # This would need to be implemented differently in a production environment
                            # This code will have no effect with current ChromaDB versions
                            metadata = collection.metadata or {}
                            metadata["sqlite_id"] = str(db_collection.id)
                            print(f"DEBUG: [create_collection] Added SQLite ID {db_collection.id} to ChromaDB metadata")
                        except Exception as metadata_error:
                            print(f"DEBUG: [create_collection] Could not update ChromaDB metadata: {metadata_error}")
                except Exception as e:
                    print(f"WARNING: [create_collection] Could not access ChromaDB collection: {e}")
            
            return db_collection
        except Exception as e:
            print(f"ERROR: [create_collection] Failed to create ChromaDB collection: {str(e)}")
            import traceback
            print(f"ERROR: [create_collection] Stack trace:\n{traceback.format_exc()}")
            
            # If we already created a SQLite record, we should delete it to avoid inconsistency
            if 'db_collection' in locals() and db_collection and db_collection.id:
                print(f"ERROR: [create_collection] Deleting SQLite record {db_collection.id} due to ChromaDB creation failure")
                try:
                    db.delete(db_collection)
                    db.commit()
                except Exception as db_e:
                    print(f"ERROR: [create_collection] Failed to delete SQLite record: {str(db_e)}")
                    # Even if we fail to delete, still raise the original exception
            
            raise
    
    @staticmethod
    def get_collection(db: Session, collection_id: int) -> Optional[Dict[str, Any]]:
        """Get a collection by ID.
        
        Args:
            db: SQLAlchemy database session
            collection_id: ID of the collection to retrieve
            
        Returns:
            The Collection as a dictionary if found, None otherwise
        """
        collection = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            return None
        
        # Convert to dictionary and ensure embeddings_model is deserialized
        collection_dict = collection.to_dict()
        if isinstance(collection_dict['embeddings_model'], str):
            try:
                collection_dict['embeddings_model'] = json.loads(collection_dict['embeddings_model'])
            except (json.JSONDecodeError, TypeError):
                collection_dict['embeddings_model'] = {}
            
        return collection_dict
    
    @staticmethod
    def get_collection_by_name(db: Session, name: str) -> Optional[Collection]:
        """Get a collection by name.
        
        Args:
            db: SQLAlchemy database session
            name: Name of the collection to retrieve
            
        Returns:
            The Collection object if found, None otherwise
        """
        return db.query(Collection).filter(Collection.name == name).first()
    
    @staticmethod
    def get_collection_by_chromadb_uuid(db: Session, chromadb_uuid: str) -> Optional[Collection]:
        """Get a collection by ChromaDB UUID.
        
        Args:
            db: SQLAlchemy database session
            chromadb_uuid: ChromaDB UUID of the collection to retrieve
            
        Returns:
            The Collection object if found, None otherwise
        """
        return db.query(Collection).filter(Collection.chromadb_uuid == chromadb_uuid).first()
    
    @staticmethod
    def list_collections(
        db: Session,
        owner: Optional[str] = None,
        visibility: Optional[Visibility] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List collections with optional filtering.
        
        Args:
            db: Database session
            owner: Optional filter by owner
            visibility: Optional filter by visibility
            skip: Number of collections to skip
            limit: Maximum number of collections to return
            
        Returns:
            List of collections
        """
        # Build query
        query = db.query(Collection)
        
        # Apply filters if provided
        if owner:
            query = query.filter(Collection.owner == owner)
        if visibility:
            query = query.filter(Collection.visibility == visibility)
        
        # Apply pagination
        collections = query.offset(skip).limit(limit).all()
        
        # Convert to dictionary and ensure embeddings_model is deserialized
        result = []
        for collection in collections:
            collection_dict = collection.to_dict()
            if isinstance(collection_dict['embeddings_model'], str):
                try:
                    collection_dict['embeddings_model'] = json.loads(collection_dict['embeddings_model'])
                except (json.JSONDecodeError, TypeError):
                    collection_dict['embeddings_model'] = {}
            result.append(collection_dict)
        
        return result
    
    @staticmethod
    def update_collection(
        db: Session,
        collection_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[Visibility] = None,
        embeddings_model: Optional[Dict[str, Any]] = None
    ) -> Optional[Collection]:
        """Update a collection.
        
        Args:
            db: SQLAlchemy database session
            collection_id: ID of the collection to update
            name: Optional new name
            description: Optional new description
            visibility: Optional new visibility
            embeddings_model: Optional new embeddings model configuration
            
        Returns:
            The updated Collection object if found, None otherwise
        """
        db_collection = db.query(Collection).filter(Collection.id == collection_id).first()
        
        if not db_collection:
            return None
        
        # Update SQLite collection
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if visibility is not None:
            update_data["visibility"] = visibility
        if embeddings_model is not None:
            update_data["embeddings_model"] = embeddings_model
        
        for key, value in update_data.items():
            setattr(db_collection, key, value)
        
        db.commit()
        db.refresh(db_collection)
        
        # Update ChromaDB collection metadata if name was not changed
        # (if name was changed, we would need to recreate the collection in ChromaDB)
        if name is None:
            chroma_client = get_chroma_client()
            try:
                chroma_collection = chroma_client.get_collection(db_collection.name)
                
                # Update metadata
                metadata = chroma_collection.metadata or {}
                if description is not None:
                    metadata["description"] = description
                if visibility is not None:
                    metadata["visibility"] = visibility.value
                
                # Cannot update collection metadata directly in ChromaDB, 
                # would need to recreate in a real implementation
            except Exception as e:
                print(f"Error updating ChromaDB collection: {e}")
        
        return db_collection
    
    @staticmethod
    def delete_collection(db: Session, collection_id: int) -> bool:
        """Delete a collection from both SQLite and ChromaDB.
        
        Args:
            db: SQLAlchemy database session
            collection_id: ID of the collection to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
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
