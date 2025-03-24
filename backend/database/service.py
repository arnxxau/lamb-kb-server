"""
Database service module for the Lamb Knowledge Base Server.

This module provides service functions for interacting with the SQLite and ChromaDB databases.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from sqlalchemy.orm import Session

from .models import Collection, Visibility
from .connection import get_chroma_client, get_embedding_function


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
        try:
            # Default values if embeddings_model is None or empty
            vendor = "local"
            model = "sentence-transformers/all-MiniLM-L6-v2"
            api_key = None
            
            import os
            
            if embeddings_model:
                # Resolve 'default' values to actual values from environment
                resolved_config = {}
                
                vendor = embeddings_model.get("vendor")
                if vendor == "default":
                    vendor = os.getenv("EMBEDDINGS_VENDOR")
                    if not vendor:
                        raise ValueError("EMBEDDINGS_VENDOR environment variable not set but 'default' specified")
                resolved_config["vendor"] = vendor
                
                model = embeddings_model.get("model")
                if model == "default":
                    model = os.getenv("EMBEDDINGS_MODEL")
                    if not model:
                        raise ValueError("EMBEDDINGS_MODEL environment variable not set but 'default' specified")
                resolved_config["model"] = model
                
                # API key is optional (empty string is valid)
                api_key = embeddings_model.get("apikey")
                if api_key == "default":
                    api_key = os.getenv("EMBEDDINGS_APIKEY", "")
                resolved_config["apikey"] = api_key
                
                # API endpoint (required for some vendors like Ollama)
                api_endpoint = embeddings_model.get("api_endpoint")
                if api_endpoint == "default":
                    api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
                    if not api_endpoint and vendor == "ollama":
                        raise ValueError("EMBEDDINGS_ENDPOINT environment variable not set but 'default' specified for Ollama")
                if api_endpoint:  # Only add if not None
                    resolved_config["api_endpoint"] = api_endpoint
                
                # Replace the original config with resolved values
                embeddings_model = resolved_config
                
                print(f"DEBUG: [create_collection] Resolved embeddings config: {embeddings_model}")
                
            # NOW create the SQLite record with RESOLVED embeddings_model values
            db_collection = Collection(
                name=name,
                owner=owner,
                description=description,
                visibility=visibility,
                embeddings_model=json.dumps(embeddings_model) if isinstance(embeddings_model, dict) else embeddings_model
            )
            
            db.add(db_collection)
            db.commit()
            db.refresh(db_collection)
            
            # Get the embedding function directly with parameters since collection doesn't exist yet
            from database.connection import get_embedding_function_by_params
            
            # Use the resolved values from embeddings_model, not the original variables
            embedding_func = get_embedding_function_by_params(
                vendor=embeddings_model.get("vendor"),
                model_name=embeddings_model.get("model"),
                api_key=embeddings_model.get("apikey", ""),
                api_endpoint=embeddings_model.get("api_endpoint")
            )
        except ValueError as e:
            # Log the error but continue with default embeddings
            print(f"Warning: Could not create custom embedding function: {str(e)}")
            print("Using default embedding function instead")
            embedding_func = None
        
        # Create the collection with the appropriate embedding function
        collection_params = {
            "name": name,
            "metadata": {
                "owner": owner,
                "description": description,
                "visibility": visibility.value,
                "sqlite_id": db_collection.id,
                "creation_date": datetime.utcnow().isoformat(),
                "embeddings_model": json.dumps(embeddings_model, ensure_ascii=False)
            }
            # Make sure we have a valid, non-empty name
            # ChromaDB has strict naming requirements
        }
        
        # Add embedding function if available
        if embedding_func:
            # Test the embedding function with a simple string to make sure it works
            try:
                print(f"DEBUG: [create_collection] Testing embedding function")
                test_result = embedding_func(["Test string for embedding"])
                print(f"DEBUG: [create_collection] Embedding function test successful. Dimension: {len(test_result[0])}")
                collection_params["embedding_function"] = embedding_func
            except Exception as e:
                print(f"ERROR: [create_collection] Embedding function test failed: {str(e)}")
                print(f"ERROR: [create_collection] Falling back to default embedding function")
                # Don't add embedding_function to params - let ChromaDB use its default
        
        # Add detailed debugging for collection creation
        print(f"DEBUG: [create_collection] Creating ChromaDB collection with name: {name}")
        print(f"DEBUG: [create_collection] Collection parameters: {collection_params}")
        
        try:
            chroma_collection = chroma_client.create_collection(**collection_params)
            print(f"DEBUG: [create_collection] Successfully created ChromaDB collection")
        except Exception as e:
            print(f"ERROR: [create_collection] Failed to create ChromaDB collection: {str(e)}")
            import traceback
            print(f"ERROR: [create_collection] Stack trace:\n{traceback.format_exc()}")
            raise
        
        return db_collection
    
    @staticmethod
    def get_collection(db: Session, collection_id: int) -> Optional[Collection]:
        """Get a collection by ID.
        
        Args:
            db: SQLAlchemy database session
            collection_id: ID of the collection to retrieve
            
        Returns:
            The Collection object if found, None otherwise
        """
        return db.query(Collection).filter(Collection.id == collection_id).first()
    
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
    def list_collections(
        db: Session, 
        owner: Optional[str] = None,
        visibility: Optional[Visibility] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Collection]:
        """List collections with optional filtering.
        
        Args:
            db: SQLAlchemy database session
            owner: Optional filter by owner
            visibility: Optional filter by visibility
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of Collection objects
        """
        query = db.query(Collection)
        
        if owner:
            query = query.filter(Collection.owner == owner)
        
        if visibility:
            query = query.filter(Collection.visibility == visibility)
        
        return query.offset(skip).limit(limit).all()
    
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
