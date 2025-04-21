"""
Document ingestion service.

This module provides services for ingesting documents into collections using various plugins.
"""

import os
import shutil
import uuid
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, BinaryIO, Union

import chromadb
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from database.connection import get_chroma_client, get_embedding_function
from database.models import Collection, FileRegistry, FileStatus
from database.service import CollectionRepository
from database.ingestion import IngestionRepository
from plugins.base import PluginRegistry, IngestPlugin


class IngestionService:
    """Service for ingesting documents into collections."""
    
    # Base directory for storing uploaded files
    STATIC_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "static"
    
    # URL prefix for accessing static files
    STATIC_URL_PREFIX = os.getenv("HOME_URL", "http://localhost:9090") + "/static"
    
    @classmethod
    def _ensure_dirs(cls):
        """Ensure necessary directories exist."""
        cls.STATIC_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def _get_user_dir(cls, owner: str) -> Path:
        """Get the user's root directory for documents.
        
        Args:
            owner: Owner of the documents
            
        Returns:
            Path to the user's document directory
        """
        user_dir = cls.STATIC_DIR / owner
        user_dir.mkdir(exist_ok=True)
        return user_dir
    
    @classmethod
    def _get_collection_dir(cls, owner: str, collection_name: str) -> Path:
        """Get the collection directory for a user.
        
        Args:
            owner: Owner of the collection
            collection_name: Name of the collection
            
        Returns:
            Path to the collection directory
        """
        collection_dir = cls._get_user_dir(owner) / collection_name
        collection_dir.mkdir(exist_ok=True)
        return collection_dir
    
    @classmethod
    def save_uploaded_file(cls, file: UploadFile, owner: str, collection_name: str) -> Dict[str, str]:
        """Save an uploaded file to the appropriate directory.
        
        Args:
            file: The uploaded file
            owner: Owner of the collection
            collection_name: Name of the collection
            
        Returns:
            Dictionary with file path and URL
        """
        print(f"DEBUG: [save_uploaded_file] Starting to save file: {file.filename if file else 'None'}")
        cls._ensure_dirs()
        
        # Create a unique filename to avoid collisions
        original_filename = file.filename or "unknown"
        file_extension = original_filename.split(".")[-1] if "." in original_filename else ""
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}" if file_extension else f"{uuid.uuid4().hex}"
        print(f"DEBUG: [save_uploaded_file] Generated unique filename: {unique_filename}")
        
        # Store original filename as part of the metadata
        sanitized_original_name = os.path.basename(original_filename)
        
        # Get the collection directory and prepare the file path
        collection_dir = cls._get_collection_dir(owner, collection_name)
        file_path = collection_dir / unique_filename
        print(f"DEBUG: [save_uploaded_file] File will be saved to: {file_path}")
        
        try:
            # Save the file
            print(f"DEBUG: [save_uploaded_file] Starting file copy operation")
            with open(file_path, "wb") as f:
                # Read in chunks to avoid memory issues with large files
                print(f"DEBUG: [save_uploaded_file] Reading file content")
                content = file.file.read()
                print(f"DEBUG: [save_uploaded_file] File content read, size: {len(content)} bytes")
                f.write(content)
            print(f"DEBUG: [save_uploaded_file] File saved successfully")
            
            # Create URL path for the file
            relative_path = file_path.relative_to(cls.STATIC_DIR)
            file_url = f"{cls.STATIC_URL_PREFIX}/{relative_path}"
            
            result = {
                "file_path": str(file_path),
                "file_url": file_url,
                "original_filename": sanitized_original_name
            }
            print(f"DEBUG: [save_uploaded_file] File saved, returning result")
            return result
        except Exception as e:
            print(f"DEBUG: [save_uploaded_file] ERROR saving file: {str(e)}")
            import traceback
            print(f"DEBUG: [save_uploaded_file] Stack trace:\n{traceback.format_exc()}")
            raise
    
    @classmethod
    def register_file(cls, 
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
                     status: FileStatus = FileStatus.COMPLETED) -> FileRegistry:
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
        return IngestionRepository.register_file(
            db=db,
            collection_id=collection_id,
            file_path=file_path,
            file_url=file_url,
            original_filename=original_filename,
            plugin_name=plugin_name,
            plugin_params=plugin_params,
            owner=owner,
            document_count=document_count,
            content_type=content_type,
            status=status
        )
    
    @classmethod
    def update_file_status(cls, db: Session, file_id: int, status: FileStatus) -> FileRegistry:
        """Update the status of a file in the registry.
        
        Args:
            db: Database session
            file_id: ID of the file registry entry
            status: New status
            
        Returns:
            The updated FileRegistry entry or None if not found
        """
        return IngestionRepository.update_file_status(db, file_id, status)
        
    @classmethod
    def list_plugins(cls) -> List[Dict[str, Any]]:
        """List all available ingestion plugins.
        
        Returns:
            List of plugin metadata
        """
        return PluginRegistry.list_plugins()
    
    @classmethod
    def get_plugin(cls, plugin_name: str) -> Optional[IngestPlugin]:
        """Get a plugin by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance if found, None otherwise
        """
        plugin_class = PluginRegistry.get_plugin(plugin_name)
        if plugin_class:
            return plugin_class()
        return None
    
    @classmethod
    def get_file_url(cls, file_path: str) -> str:
        """Get the URL for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            URL for accessing the file
        """
        try:
            file_path_obj = Path(file_path)
            relative_path = file_path_obj.relative_to(cls.STATIC_DIR)
            return f"{cls.STATIC_URL_PREFIX}/{relative_path}"
        except ValueError:
            # If file is not under STATIC_DIR, it's not accessible via URL
            return ""
    
    @classmethod
    def ingest_file(
        cls, 
        file_path: str,
        plugin_name: str,
        plugin_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ingest a file using the specified plugin.
        
        Args:
            file_path: Path to the file to ingest
            plugin_name: Name of the plugin to use
            plugin_params: Parameters for the plugin
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            HTTPException: If the plugin is not found or ingestion fails
        """
        print(f"DEBUG: [ingest_file] Starting ingestion for file: {file_path}")
        print(f"DEBUG: [ingest_file] Using plugin: {plugin_name}")
        print(f"DEBUG: [ingest_file] Plugin params: {plugin_params}")
        
        # Check file exists
        if not os.path.exists(file_path):
            print(f"DEBUG: [ingest_file] ERROR: File does not exist: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"DEBUG: [ingest_file] File size: {file_size} bytes")
        
        plugin = cls.get_plugin(plugin_name)
        if not plugin:
            print(f"DEBUG: [ingest_file] ERROR: Plugin {plugin_name} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Ingestion plugin '{plugin_name}' not found"
            )
        
        print(f"DEBUG: [ingest_file] Plugin found: {plugin_name}")
        
        try:
            # Get file URL for the document
            file_url = cls.get_file_url(file_path)
            print(f"DEBUG: [ingest_file] File URL: {file_url}")
            
            # Add file_url to plugin params
            plugin_params_with_url = plugin_params.copy()
            plugin_params_with_url["file_url"] = file_url
            
            # Ingest the file with the plugin
            print(f"DEBUG: [ingest_file] Calling plugin.ingest()")
            documents = plugin.ingest(file_path, **plugin_params_with_url)
            print(f"DEBUG: [ingest_file] Plugin returned {len(documents)} chunks")
            
            return documents
            
        except Exception as e:
            print(f"DEBUG: [ingest_file] ERROR during ingestion: {str(e)}")
            import traceback
            print(f"DEBUG: [ingest_file] Stack trace:\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to ingest file: {str(e)}"
            )
    
    @classmethod
    def add_documents_to_collection(
        cls,
        db: Session,
        collection_id: int,
        documents: List[Dict[str, Any]],
        embeddings_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Add documents to a collection.
        
        This method delegates to IngestionRepository for database operations.
        
        Args:
            db: Database session
            collection_id: ID of the collection
            documents: List of document chunks with metadata
            embeddings_function: Optional custom embeddings function
            
        Returns:
            Status information about the ingestion
            
        Raises:
            HTTPException: If errors occur during ingestion
        """
        try:
            return IngestionRepository.add_documents_to_collection(
                db=db, 
                collection_id=collection_id, 
                documents=documents,
                embeddings_function=embeddings_function
            )
        except ValueError as e:
            # Convert ValueError to HTTPException for API responses
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )