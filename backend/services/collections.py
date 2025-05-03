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

logger = logging.getLogger(__name__)


class CollectionsService:
    """Service for handling collection-related API endpoints."""
    
    @staticmethod
    def get_database_status():
        return CollectionRepository.get_database_status()
    
    @staticmethod
    def _resolve_embeddings_model(embeddings_model: EmbeddingsModel) -> Dict[str, Any]:
        if not embeddings_model:
            return {}
            
        model_info = embeddings_model.model_dump()
        resolved_config = {}
        
        vendor = model_info.get("vendor")
        if vendor == "default":
            vendor = os.getenv("EMBEDDINGS_VENDOR")
            if not vendor:
                raise ConfigurationException(
                    "EMBEDDINGS_VENDOR environment variable not set but 'default' specified"
                )
        resolved_config["vendor"] = vendor
        
        model = model_info.get("model")
        if model == "default":
            model = os.getenv("EMBEDDINGS_MODEL")
            if not model:
                raise ConfigurationException(
                    "EMBEDDINGS_MODEL environment variable not set but 'default' specified"
                )
        resolved_config["model"] = model
        
        api_key = model_info.get("apikey")
        if api_key == "default":
            api_key = os.getenv("EMBEDDINGS_APIKEY", "")
    
        if vendor == "openai":
            logger.info(f"Using OpenAI API key: {'[PROVIDED]' if api_key else '[MISSING]'}")
            
        resolved_config["apikey"] = api_key
        
        api_endpoint = model_info.get("api_endpoint")
        if api_endpoint == "default":
            api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
            if not api_endpoint and vendor == "ollama":
                raise ConfigurationException(
                    "EMBEDDINGS_ENDPOINT environment variable not set but 'default' specified for Ollama"
                )
        if api_endpoint:
            resolved_config["api_endpoint"] = api_endpoint
            
        logger.info(f"Resolved embeddings config: {resolved_config}")
        
        return resolved_config
    
    @classmethod
    def create_collection(
        cls,
        collection: "CollectionCreate",
    ) -> Dict[str, Any]:
        visibility_enum = Visibility(collection.visibility)

        embeddings_model = cls._resolve_embeddings_model(collection.embeddings_model or {})
        embedding_function = get_embedding_function_by_params(
            vendor=embeddings_model.get("vendor"),
            model_name=embeddings_model.get("model"),
            api_key=embeddings_model.get("apikey", ""),
            api_endpoint=embeddings_model.get("api_endpoint", "")
        )

        _ = embedding_function(["validation test"])

        return CollectionRepository.create_collection(
            name=collection.name,
            owner=collection.owner,
            description=collection.description,
            visibility=visibility_enum,
            embeddings_model=embeddings_model,
            embedding_function=embedding_function
        )
    
    @staticmethod
    def list_collections(
        skip: int = 0,
        limit: int = 100,
        owner: Optional[str] = None,
        visibility: Optional[str] = None
    ) -> Dict[str, Any]:
        visibility_enum = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise ValidationException(
                    f"Invalid visibility value: {visibility}. Must be 'private' or 'public'."
                )
        
        collections = CollectionRepository.list_collections(
            owner=owner,
            visibility=visibility_enum,
            skip=skip,
            limit=limit
        )
        
        total = len(collections)
        
        return {
            "total": total,
            "items": collections
        }
    
    @staticmethod
    def get_collection(
        collection_id: int
    ) -> Dict[str, Any]:
        collection = CollectionRepository.get_collection(collection_id)
        if not collection:
            raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
        return collection
    
    @staticmethod
    def list_files(
        collection_id: int,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if status:
            try:
                FileStatus(status)
            except ValueError:
                raise ValidationException(
                    f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
                )

        return CollectionRepository.list_files(collection_id, status)
    
    @staticmethod
    def update_file_status(
        file_id: int,
        status: str
    ) -> Dict[str, Any]:
        try:
            file_status = FileStatus(status)
        except ValueError:
            raise ValidationException(
                f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
            )
        
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
        visibility_enum: Optional[Visibility] = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise ValidationException(
                    f"Invalid visibility value: {visibility}. Must be 'private' or 'public'."
                )

        existing = CollectionRepository.get_collection(collection_id)
        if not existing:
            raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")

        current_conf = existing.embeddings_model or {}
        new_conf = current_conf.copy()
        if endpoint is not None:
            new_conf['api_endpoint'] = endpoint
        if apikey is not None:
            new_conf['apikey'] = apikey

        existing_model = current_conf.get('model')
        existing_vendor = current_conf.get('vendor')
        if model is not None and model != existing_model:
            logger.warning(
                f"Changing embeddings model from '{existing_model}' to '{model}' is not supported and will be ignored."
            )
        if vendor is not None and vendor != existing_vendor:
            logger.warning(
                f"Changing embeddings vendor from '{existing_vendor}' to '{vendor}' is not supported and will be ignored."
            )

        old_name = existing.name
        rename_from = old_name if name and name != old_name else None
        rename_to = name if name and name != old_name else None

        updated = CollectionRepository.update_collection(
            collection_id=collection_id,
            name=name,
            description=description,
            visibility=visibility_enum,
            embeddings_model=new_conf,
            rename_from=rename_from,
            rename_to=rename_to
        )
        if not updated:
            raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
            
        return updated