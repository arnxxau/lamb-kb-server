"""
Collections service module for handling collection-related endpoint logic.

This module provides service functions for handling collection-related API endpoints,
separating the business logic from the FastAPI route definitions.
"""

import json
import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database.models import Collection, Visibility, FileRegistry, FileStatus
from database.service import CollectionRepository
from database.ingestion import IngestionRepository
from services.ingestion import IngestionService
from schemas.collection import (
    CollectionCreate, 
    CollectionUpdate, 
    CollectionResponse, 
    CollectionList
)
from database.connection import get_embedding_function


class CollectionsService:
    """Service for handling collection-related API endpoints."""
    
    @staticmethod
    def create_collection(
        collection: CollectionCreate,
    ) -> Dict[str, Any]:
        """Create a new knowledge base collection.
        
        Args:
            collection: Collection data from request body with resolved default values
            
        Returns:
            The created collection
            
        Raises:
            HTTPException: If collection creation fails
        """
        # Check if collection with this name already exists
        existing = CollectionRepository.get_collection_by_name(collection.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection with name '{collection.name}' already exists"
            )
        
        # Convert visibility string to enum
        try:
            visibility = Visibility(collection.visibility)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid visibility value: {collection.visibility}. Must be 'private' or 'public'."
            )
        
        # Create the collection
        try:
            # Handle the embeddings model configuration
            embeddings_model = {}
            if collection.embeddings_model:
                # Get the model values from the request
                # Note: Default values should already be resolved by main.py
                model_info = collection.embeddings_model.model_dump()
                
                # We'll still validate the embeddings model configuration
                try:
                    # Create a temporary DB collection record for validation
                    from database.models import Collection
                    temp_collection = Collection(id=-1, name="temp_validation", 
                                              owner="system", description="Validation only", 
                                              embeddings_model=model_info)
                    
                    print(f"DEBUG: [create_collection] Validating {model_info.get('vendor')} embeddings with model: {model_info.get('model')}")
                    
                    # Try to create an embedding function with this configuration
                    # This will validate if the embeddings model configuration is valid
                    embedding_function = get_embedding_function(temp_collection)
                    
                    # Test the embedding function with a simple text
                    test_result = embedding_function(["Test embedding validation"])
                    print(f"INFO: Embeddings validation successful, dimensions: {len(test_result[0])}")

                except Exception as emb_error:
                    print(f"ERROR: Embeddings model validation failed: {str(emb_error)}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Embeddings model validation failed: {str(emb_error)}. Please check your configuration."
                    )
                
                embeddings_model = model_info
            
            # Create the collection in both databases
            db_collection = CollectionRepository.create_collection(
                name=collection.name,
                owner=collection.owner,
                description=collection.description,
                visibility=visibility,
                embeddings_model=embeddings_model
            )

            # Verify the collection was created successfully in both databases
            if not db_collection.chromadb_uuid:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Collection was created but ChromaDB UUID was not stored"
                )
            
            return db_collection
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create collection: {str(e)}"
            )
    
    @staticmethod
    def list_collections(
        skip: int = 0,
        limit: int = 100,
        owner: Optional[str] = None,
        visibility: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all available knowledge base collections with optional filtering.
        
        Args:
            skip: Number of collections to skip
            limit: Maximum number of collections to return
            owner: Optional filter by owner
            visibility: Optional filter by visibility
            
        Returns:
            Dict with total count and list of collections
            
        Raises:
            HTTPException: If invalid visibility value is provided
        """
        # Convert visibility string to enum if provided
        visibility_enum = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid visibility value: {visibility}. Must be 'private' or 'public'."
                )
        
        # Get collections with filtering
        collections = CollectionRepository.list_collections(
            owner=owner,
            visibility=visibility_enum,
            skip=skip,
            limit=limit
        )
        
        # Get total count from repository
        total = len(collections)
        
        return {
            "total": total,
            "items": collections
        }
    
    @staticmethod
    def get_collection(
        collection_id: int
    ) -> Dict[str, Any]:
        """Get details of a specific knowledge base collection.
        
        Args:
            collection_id: ID of the collection to retrieve
            
        Returns:
            Collection details
            
        Raises:
            HTTPException: If collection not found
        """
        collection = CollectionRepository.get_collection(collection_id)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection with ID {collection_id} not found"
            )
        return collection
    
    @staticmethod
    def list_files(
        collection_id: int,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all files in a collection.
        
        Args:
            collection_id: ID of the collection
            status: Optional filter by status
            
        Returns:
            List of file registry entries
            
        Raises:
            HTTPException: If collection not found or status invalid
        """
        # Check if collection exists
        collection = CollectionRepository.get_collection(collection_id)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection with ID {collection_id} not found"
            )
        
        # Validate status if provided
        if status:
            try:
                FileStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
                )
        
        # Get files from repository
        return CollectionRepository.list_files(collection_id, status)
    
    @staticmethod
    def update_file_status(
        file_id: int,
        status: str
    ) -> Dict[str, Any]:
        """Update the status of a file in the registry.
        
        Args:
            file_id: ID of the file registry entry
            status: New status
            
        Returns:
            Updated file registry entry
            
        Raises:
            HTTPException: If file not found or status invalid
        """
        # Validate status
        try:
            file_status = FileStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
            )
        
        # Update file status
        result = CollectionRepository.update_file_status(file_id, file_status)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found"
            )
        
        return result