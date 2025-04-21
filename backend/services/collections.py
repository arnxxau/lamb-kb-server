"""
Collections service module for handling collection-related endpoint logic.

This module provides service functions for handling collection-related API endpoints,
separating the business logic from the FastAPI route definitions.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import status, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from repository.models import Collection, Visibility, FileRegistry, FileStatus
from repository.collections import CollectionRepository
from repository.ingestion import IngestionRepository
from services.ingestion import IngestionService
from schemas.collection import (
    CollectionCreate, 
    CollectionUpdate, 
    CollectionResponse, 
    CollectionList,
    EmbeddingsModel
)
from repository.connection import get_embedding_function
from exceptions import (
    ResourceNotFoundException, 
    ValidationException, 
    ResourceAlreadyExistsException,
    ConfigurationException,
    ProcessingException
)

# Set up logging
logger = logging.getLogger(__name__)


class CollectionsService:
    """Service for handling collection-related API endpoints."""
    
    @staticmethod
    def resolve_embeddings_model(embeddings_model: EmbeddingsModel) -> Dict[str, Any]:
        """Resolve default values in embeddings model configuration.
        
        Args:
            embeddings_model: The embeddings model configuration from the request
            
        Returns:
            Resolved embeddings model configuration
            
        Raises:
            ConfigurationException: If default values cannot be resolved
        """
        if not embeddings_model:
            return {}
            
        model_info = embeddings_model.model_dump()
        resolved_config = {}
        
        # Resolve vendor
        vendor = model_info.get("vendor")
        if vendor == "default":
            vendor = os.getenv("EMBEDDINGS_VENDOR")
            if not vendor:
                raise ConfigurationException(
                    "EMBEDDINGS_VENDOR environment variable not set but 'default' specified"
                )
        resolved_config["vendor"] = vendor
        
        # Resolve model
        model = model_info.get("model")
        if model == "default":
            model = os.getenv("EMBEDDINGS_MODEL")
            if not model:
                raise ConfigurationException(
                    "EMBEDDINGS_MODEL environment variable not set but 'default' specified"
                )
        resolved_config["model"] = model
        
        # Resolve API key (optional)
        api_key = model_info.get("apikey")
        if api_key == "default":
            api_key = os.getenv("EMBEDDINGS_APIKEY", "")
        
        # Only log whether we have a key or not, never log the key itself or its contents
        if vendor == "openai":
            logger.info(f"Using OpenAI API key: {'[PROVIDED]' if api_key else '[MISSING]'}")
            
        resolved_config["apikey"] = api_key
        
        # Resolve API endpoint (needed for some vendors like Ollama)
        api_endpoint = model_info.get("api_endpoint")
        if api_endpoint == "default":
            api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
            if not api_endpoint and vendor == "ollama":
                raise ConfigurationException(
                    "EMBEDDINGS_ENDPOINT environment variable not set but 'default' specified for Ollama"
                )
        if api_endpoint:  # Only add if not None
            resolved_config["api_endpoint"] = api_endpoint
            
        # Log the resolved configuration
        logger.info(f"Resolved embeddings config: {resolved_config}")
        
        return resolved_config
    
    @staticmethod
    def create_collection(
        collection: CollectionCreate,
    ) -> Dict[str, Any]:
        """Create a new knowledge base collection.
        
        Args:
            collection: Collection data from request body
            
        Returns:
            The created collection
            
        Raises:
            ResourceAlreadyExistsException: If collection with same name already exists
            ValidationException: If visibility value is invalid
            ConfigurationException: If embeddings model configuration is invalid
            ProcessingException: If collection creation fails
        """
        # Check if collection with this name already exists
        existing = CollectionRepository.get_collection_by_name(collection.name)
        if existing:
            raise ResourceAlreadyExistsException(
                f"Collection with name '{collection.name}' already exists"
            )
        
        # Convert visibility string to enum
        try:
            visibility = Visibility(collection.visibility)
        except ValueError:
            raise ValidationException(
                f"Invalid visibility value: {collection.visibility}. Must be 'private' or 'public'."
            )
        
        # Create the collection
        try:
            # Handle the embeddings model configuration
            embeddings_model = {}
            if collection.embeddings_model:
                # Resolve default values
                resolved_config = CollectionsService.resolve_embeddings_model(collection.embeddings_model)
                
                # We'll still validate the embeddings model configuration
                try:
                    # Create a temporary DB collection record for validation
                    temp_collection = Collection(id=-1, name="temp_validation", 
                                              owner="system", description="Validation only", 
                                              embeddings_model=resolved_config)
                    
                    logger.debug(f"Validating {resolved_config.get('vendor')} embeddings with model: {resolved_config.get('model')}")
                    
                    # Try to create an embedding function with this configuration
                    # This will validate if the embeddings model configuration is valid
                    embedding_function = get_embedding_function(temp_collection)
                    
                    # Test the embedding function with a simple text
                    test_result = embedding_function(["Test embedding validation"])
                    logger.info(f"Embeddings validation successful, dimensions: {len(test_result[0])}")

                except Exception as emb_error:
                    logger.error(f"Embeddings model validation failed: {str(emb_error)}")
                    raise ConfigurationException(
                        f"Embeddings model validation failed: {str(emb_error)}. Please check your configuration."
                    )
                
                embeddings_model = resolved_config
            
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
                raise ProcessingException(
                    "Collection was created but ChromaDB UUID was not stored"
                )
            
            return db_collection
        except (ResourceAlreadyExistsException, ValidationException, ConfigurationException) as e:
            # Re-raise domain exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected errors in ProcessingException
            logger.error(f"Failed to create collection: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to create collection: {str(e)}")
    
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
            ValidationException: If invalid visibility value is provided
        """
        # Convert visibility string to enum if provided
        visibility_enum = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise ValidationException(
                    f"Invalid visibility value: {visibility}. Must be 'private' or 'public'."
                )
        
        # Get collections with filtering
        try:
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
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to list collections: {str(e)}")
    
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
            ResourceNotFoundException: If collection not found
        """
        collection = CollectionRepository.get_collection(collection_id)
        if not collection:
            raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
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
            ResourceNotFoundException: If collection not found
            ValidationException: If status is invalid
        """
        # Check if collection exists
        collection = CollectionRepository.get_collection(collection_id)
        if not collection:
            raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
        
        # Validate status if provided
        if status:
            try:
                FileStatus(status)
            except ValueError:
                raise ValidationException(
                    f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
                )
        
        # Get files from repository
        try:
            return CollectionRepository.list_files(collection_id, status)
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to list files: {str(e)}")
    
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
            ResourceNotFoundException: If file not found
            ValidationException: If status is invalid
        """
        # Validate status
        try:
            file_status = FileStatus(status)
        except ValueError:
            raise ValidationException(
                f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
            )
        
        # Update file status
        result = CollectionRepository.update_file_status(file_id, file_status)
        if not result:
            raise ResourceNotFoundException(f"File with ID {file_id} not found")
        
        return result
        
    @staticmethod
    def update_collection(
        collection_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        model: Optional[str] = None,
        vendor: Optional[str] = None,
        endpoint: Optional[str] = None,
        apikey: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a collection's details.
        
        Args:
            collection_id: ID of the collection to update
            name: New collection name
            description: New description
            visibility: New visibility setting ('private' or 'public')
            model: New embeddings model name (changing is not supported)
            vendor: New embeddings vendor (changing is not supported)
            endpoint: New embeddings-model endpoint
            apikey: New embeddings-model API key
            
        Returns:
            Updated collection details
            
        Raises:
            ResourceNotFoundException: If collection not found
            ValidationException: If visibility value is invalid
            ProcessingException: If update fails
        """
        # Validate visibility if provided
        visibility_enum = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise ValidationException(
                    f"Invalid visibility value: {visibility}. Must be 'private' or 'public'."
                )
        
        try:
            # First get the existing collection to check model/vendor
            existing_collection = CollectionRepository.get_collection(collection_id)
            if not existing_collection:
                raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
            
            # Handle embeddings model updates with business rules
            embeddings_model_updates = {}
            if endpoint is not None:
                embeddings_model_updates['api_endpoint'] = endpoint
            if apikey is not None:
                embeddings_model_updates['apikey'] = apikey
            
            # Check if trying to change model or vendor
            existing_model = None
            existing_vendor = None
            if isinstance(existing_collection, dict) and 'embeddings_model' in existing_collection:
                existing_model = existing_collection['embeddings_model'].get('model')
                existing_vendor = existing_collection['embeddings_model'].get('vendor')
            
            # Business rule: Cannot change model or vendor
            if model is not None and model != existing_model:
                logger.warning(f"Changing embeddings model from '{existing_model}' to '{model}' is not supported and will be ignored.")
                # Don't add to updates
            
            if vendor is not None and vendor != existing_vendor:
                logger.warning(f"Changing embeddings vendor from '{existing_vendor}' to '{vendor}' is not supported and will be ignored.")
                # Don't add to updates
                
            # Call repository method with filtered updates
            updated_collection = CollectionRepository.update_collection(
                collection_id=collection_id,
                name=name,
                description=description,
                visibility=visibility_enum,
                embeddings_model_updates=embeddings_model_updates
            )
            
            if not updated_collection:
                raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
                
            # Convert to dict if it's a model instance
            if hasattr(updated_collection, 'to_dict'):
                return updated_collection.to_dict()
            return updated_collection
            
        except (ResourceNotFoundException, ValidationException):
            # Re-raise domain exceptions
            raise
        except Exception as e:
            # Log the error and wrap in ProcessingException
            logger.error(f"Failed to update collection: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to update collection: {str(e)}")