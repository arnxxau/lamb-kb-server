import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, BinaryIO

from repository.models import FileStatus
from repository.collections import CollectionRepository
from repository.ingestion import IngestionRepository
from plugins.base import PluginRegistry, IngestPlugin

logger = logging.getLogger(__name__)

class IngestionService:
    STATIC_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "static"
    STATIC_URL_PREFIX = os.getenv("HOME_URL", "http://localhost:9090") + "/static"
    
    @classmethod
    def _ensure_dirs(cls):
        cls.STATIC_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def _get_user_dir(cls, owner: str) -> Path:
        user_dir = cls.STATIC_DIR / owner
        user_dir.mkdir(exist_ok=True)
        return user_dir
    
    @classmethod
    def _get_collection_dir(cls, owner: str, collection_name: str) -> Path:
        collection_dir = cls._get_user_dir(owner) / collection_name
        collection_dir.mkdir(exist_ok=True)
        return collection_dir
    
    @classmethod
    def save_file(cls, file_content: BinaryIO, filename: str, owner: str, 
                  collection_name: str, content_type: str = None) -> Dict[str, str]:
        cls._ensure_dirs()
        
        original_filename = filename or "unknown"
        file_extension = original_filename.split(".")[-1] if "." in original_filename else ""
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}" if file_extension else f"{uuid.uuid4().hex}"
        
        sanitized_original_name = os.path.basename(original_filename)
        
        collection_dir = cls._get_collection_dir(owner, collection_name)
        file_path = collection_dir / unique_filename
        
        with open(file_path, "wb") as f:
            content = file_content.read()
            f.write(content)
        
        relative_path = file_path.relative_to(cls.STATIC_DIR)
        file_url = f"{cls.STATIC_URL_PREFIX}/{relative_path}"
        
        return {
            "file_path": str(file_path),
            "file_url": file_url,
            "original_filename": sanitized_original_name,
            "content_type": content_type
        }
    
    @classmethod
    def list_plugins(cls) -> List[Dict[str, Any]]:
        return PluginRegistry.list_plugins()
    
    @classmethod
    def get_plugin(cls, plugin_name: str) -> Optional[IngestPlugin]:
        plugin_class = PluginRegistry.get_plugin(plugin_name)
        if plugin_class:
            return plugin_class()
        return None
    
    @classmethod
    def get_file_url(cls, file_path: str) -> str:
        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(cls.STATIC_DIR)
        return f"{cls.STATIC_URL_PREFIX}/{relative_path}"
    
    @classmethod
    def ingest_file(cls, file_path: str, plugin_name: str, plugin_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return []
        
        plugin = cls.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"Plugin {plugin_name} not found")
            return []
        
        file_url = cls.get_file_url(file_path)
        
        plugin_params_with_url = plugin_params.copy()
        plugin_params_with_url["file_url"] = file_url
        
        documents = plugin.ingest(file_path, **plugin_params_with_url)
        
        return documents
    
    @classmethod
    def get_file_content(cls, file_id: int) -> Dict[str, Any]:
        result = CollectionRepository.get_file_content(file_id)
        if not result:
            logger.error(f"File content not found for ID: {file_id}")
        return result
    
    @classmethod
    def process_uploaded_file(cls, file_path: str, plugin_name: str, plugin_params: Dict[str, Any], 
                              collection_id: int, file_registry_id: int):
        documents = cls.ingest_file(
            file_path=file_path,
            plugin_name=plugin_name, 
            plugin_params=plugin_params
        )
        
        IngestionRepository.process_file(
            file_path=file_path,
            plugin_name=plugin_name,
            params=plugin_params,
            collection_id=collection_id,
            file_registry_id=file_registry_id,
            documents=documents
        )
    
    @classmethod
    def process_urls(cls, urls: List[str], plugin_name: str, plugin_params: Dict[str, Any],
                     collection_id: int, file_registry_id: int, temp_file_path: str):
        params_with_urls = {**plugin_params, "urls": urls}
        
        documents = cls.ingest_file(
            file_path=temp_file_path,
            plugin_name=plugin_name,
            plugin_params=params_with_urls
        )
        
        IngestionRepository.process_urls(
            temp_file_path=temp_file_path,
            plugin_name=plugin_name,
            params=plugin_params,
            collection_id=collection_id,
            file_registry_id=file_registry_id,
            documents=documents
        )
        
        os.unlink(temp_file_path)