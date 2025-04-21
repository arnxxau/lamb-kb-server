"""
Document ingestion service.

This module provides services for ingesting documents into collections using various plugins.
"""

import os
import shutil
import uuid
import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, BinaryIO, Union

import chromadb
from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.orm import Session

from repository.connection import get_chroma_client, get_embedding_function, SessionLocal
from repository.models import Collection, FileRegistry, FileStatus
from repository.collections import CollectionRepository
from repository.ingestion import IngestionRepository
from plugins.base import PluginRegistry, IngestPlugin
# Removed import of CollectionsService to avoid circular import
from exceptions import (
    ResourceNotFoundException,
    ValidationException,
    ProcessingException,
    PluginNotFoundException,
    FileNotFoundException,
    ConfigurationException
)

# Set up logging
logger = logging.getLogger(__name__)


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
            
        Raises:
            ProcessingException: If saving the file fails
        """
        logger.debug(f"Starting to save file: {file.filename if file else 'None'}")
        cls._ensure_dirs()
        
        # Create a unique filename to avoid collisions
        original_filename = file.filename or "unknown"
        file_extension = original_filename.split(".")[-1] if "." in original_filename else ""
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}" if file_extension else f"{uuid.uuid4().hex}"
        logger.debug(f"Generated unique filename: {unique_filename}")
        
        # Store original filename as part of the metadata
        sanitized_original_name = os.path.basename(original_filename)
        
        # Get the collection directory and prepare the file path
        collection_dir = cls._get_collection_dir(owner, collection_name)
        file_path = collection_dir / unique_filename
        logger.debug(f"File will be saved to: {file_path}")
        
        try:
            # Save the file
            logger.debug(f"Starting file copy operation")
            with open(file_path, "wb") as f:
                # Read in chunks to avoid memory issues with large files
                logger.debug(f"Reading file content")
                content = file.file.read()
                logger.debug(f"File content read, size: {len(content)} bytes")
                f.write(content)
            logger.debug(f"File saved successfully")
            
            # Create URL path for the file
            relative_path = file_path.relative_to(cls.STATIC_DIR)
            file_url = f"{cls.STATIC_URL_PREFIX}/{relative_path}"
            
            result = {
                "file_path": str(file_path),
                "file_url": file_url,
                "original_filename": sanitized_original_name
            }
            logger.debug(f"File saved, returning result")
            return result
        except Exception as e:
            logger.error(f"ERROR saving file: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to save uploaded file: {str(e)}")
    
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
            
        Raises:
            ProcessingException: If registering the file fails
        """
        try:
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
        except Exception as e:
            logger.error(f"Failed to register file: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to register file: {str(e)}")
    
    @classmethod
    def update_file_status(cls, db: Session, file_id: int, status: FileStatus) -> FileRegistry:
        """Update the status of a file in the registry.
        
        Args:
            db: Database session
            file_id: ID of the file registry entry
            status: New status
            
        Returns:
            The updated FileRegistry entry or None if not found
            
        Raises:
            ResourceNotFoundException: If the file is not found
            ProcessingException: If updating the file status fails
        """
        try:
            result = IngestionRepository.update_file_status(db, file_id, status)
            if not result:
                raise ResourceNotFoundException(f"File with ID {file_id} not found")
            return result
        except ResourceNotFoundException:
            # Re-raise resource not found exceptions
            raise
        except Exception as e:
            logger.error(f"Failed to update file status: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to update file status: {str(e)}")
        
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
            FileNotFoundException: If the file does not exist
            PluginNotFoundException: If the plugin is not found
            ProcessingException: If ingestion fails
        """
        logger.debug(f"Starting ingestion for file: {file_path}")
        logger.debug(f"Using plugin: {plugin_name}")
        logger.debug(f"Plugin params: {plugin_params}")
        
        # Check file exists
        if not os.path.exists(file_path):
            logger.error(f"ERROR: File does not exist: {file_path}")
            raise FileNotFoundException(f"File not found: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        logger.debug(f"File size: {file_size} bytes")
        
        plugin = cls.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"ERROR: Plugin {plugin_name} not found")
            raise PluginNotFoundException(f"Ingestion plugin '{plugin_name}' not found")
        
        logger.debug(f"Plugin found: {plugin_name}")
        
        try:
            # Get file URL for the document
            file_url = cls.get_file_url(file_path)
            logger.debug(f"File URL: {file_url}")
            
            # Add file_url to plugin params
            plugin_params_with_url = plugin_params.copy()
            plugin_params_with_url["file_url"] = file_url
            
            # Ingest the file with the plugin
            logger.debug(f"Calling plugin.ingest()")
            documents = plugin.ingest(file_path, **plugin_params_with_url)
            logger.debug(f"Plugin returned {len(documents)} chunks")
            
            return documents
            
        except Exception as e:
            logger.error(f"ERROR during ingestion: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to ingest file: {str(e)}")
    
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
            ResourceNotFoundException: If collection not found
            ProcessingException: If adding documents fails
        """
        try:
            return IngestionRepository.add_documents_to_collection(
                db=db, 
                collection_id=collection_id, 
                documents=documents,
                embeddings_function=embeddings_function
            )
        except Exception as e:
            logger.error(f"Failed to add documents to collection: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to add documents to collection: {str(e)}")
    
    @classmethod
    def get_file_content(cls, file_id: int, db: Session) -> Dict[str, Any]:
        """Get the content of a file from file registry.
        
        Args:
            file_id: ID of the file registry entry
            db: Database session
            
        Returns:
            Content of the file with metadata
            
        Raises:
            ResourceNotFoundException: If file or collection not found
            FileNotFoundException: If content cannot be retrieved
        """
        try:
            # Use the repository to get file content
            from repository.collections import CollectionRepository
            
            try:
                result = CollectionRepository.get_file_content(file_id)
                return result
            except ValueError as e:
                # Map repository exceptions to domain exceptions
                error_message = str(e)
                if "not found" in error_message:
                    raise ResourceNotFoundException(error_message)
                else:
                    raise FileNotFoundException(error_message)
                
        except (ResourceNotFoundException, FileNotFoundException):
            # Re-raise domain exceptions
            raise
        except Exception as e:
            logger.error(f"Error retrieving file content: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to retrieve file content: {str(e)}")
    
    @classmethod
    def verify_collection_exists(cls, collection_id: int) -> Dict[str, Any]:
        """Verify that a collection exists in both SQLite and ChromaDB.
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            Collection details if it exists
            
        Raises:
            ResourceNotFoundException: If collection not found in either database
        """
        # Get collection directly from repository to avoid circular import
        collection = CollectionRepository.get_collection(collection_id)
        if not collection:
            raise ResourceNotFoundException(f"Collection with ID {collection_id} not found")
        
        # Get collection name - handle both dict-like and attribute access
        collection_name = collection['name'] if isinstance(collection, dict) else collection.name
            
        # Also verify ChromaDB collection exists
        try:
            chroma_client = get_chroma_client()
            chroma_collection = chroma_client.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Failed to get ChromaDB collection: {str(e)}", exc_info=True)
            raise ResourceNotFoundException(
                f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
            )
        
        return collection
    
    @classmethod
    def ingest_url_to_collection(
        cls,
        collection_id: int,
        urls: List[str],
        plugin_name: str,
        plugin_params: Dict[str, Any],
        db: Session,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Fetch, process, and add content from URLs to a collection.
        
        Args:
            collection_id: ID of the collection
            urls: List of URLs to ingest
            plugin_name: Name of the ingestion plugin to use
            plugin_params: Parameters for the plugin
            db: Database session
            background_tasks: FastAPI background tasks
            
        Returns:
            Status information about the ingestion operation
            
        Raises:
            ResourceNotFoundException: If collection not found
            PluginNotFoundException: If plugin not found
            ValidationException: If no URLs provided
            ProcessingException: If ingestion fails
        """
        # Verify collection exists in both SQLite and ChromaDB
        collection = cls.verify_collection_exists(collection_id)
        collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
        # Check if plugin exists
        plugin = cls.get_plugin(plugin_name)
        if not plugin:
            raise PluginNotFoundException(f"Ingestion plugin '{plugin_name}' not found")
        
        # Check if URLs are provided
        if not urls:
            raise ValidationException("No URLs provided")
        
        try:
            # Create a temporary file to track this URL ingestion
            import tempfile
            import uuid
            import os
            
            temp_dir = os.path.join(tempfile.gettempdir(), "url_ingestion")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.url")
            
            # Write URLs to the temporary file (just for tracking)
            with open(temp_file_path, "w") as f:
                for url in urls:
                    f.write(f"{url}\n")
            
            # Step 1: Register the URL ingestion in the FileRegistry with PROCESSING status
            # Store the first URL as the filename to make it easier to display and preview
            first_url = urls[0] if urls else "unknown_url"
            file_registry = cls.register_file(
                db=db,
                collection_id=collection_id,
                file_path=temp_file_path,
                file_url=first_url,  # Store the URL for direct access
                original_filename=first_url,  # Use the first URL as the original filename for better display
                plugin_name="url_ingest",  # Ensure consistent plugin name
                plugin_params={"urls": urls, **plugin_params},
                owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
                document_count=0,  # Will be updated after processing
                content_type="text/plain",
                status=FileStatus.PROCESSING  # Set initial status to PROCESSING
            )
            
            # Step 2: Schedule background task for processing and adding documents
            def process_urls_in_background(urls: List[str], plugin_name: str, params: dict, 
                                       collection_id: int, file_registry_id: int):
                try:
                    logger.debug(f"[background_task] Started processing URLs: {', '.join(urls[:3])}...")
                    
                    # Create a new session for the background task
                    db_background = SessionLocal()
                    
                    try:
                        # Make a placeholder file path for the URL ingestion
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file_path = temp_file.name
                        temp_file.close()
                        
                        # Step 2.1: Process URLs with plugin
                        # Add URLs to the plugin parameters
                        full_params = {**params, "urls": urls}
                        
                        documents = cls.ingest_file(
                            file_path=temp_file_path,  # This is just a placeholder
                            plugin_name=plugin_name,
                            plugin_params=full_params
                        )
                        
                        # Step 2.2: Add documents to collection
                        result = cls.add_documents_to_collection(
                            db=db_background,
                            collection_id=collection_id,
                            documents=documents
                        )
                        
                        # Step 2.3: Update file registry with completed status and document count
                        cls.update_file_status(
                            db=db_background, 
                            file_id=file_registry_id, 
                            status=FileStatus.COMPLETED
                        )
                        
                        # Update document count
                        file_reg = db_background.query(FileRegistry).filter(FileRegistry.id == file_registry_id).first()
                        if file_reg:
                            file_reg.document_count = len(documents)
                            db_background.commit()
                        
                        # Clean up the temporary file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                        
                        logger.debug(f"[background_task] Completed processing URLs with {len(documents)} documents")
                    finally:
                        db_background.close()
                        
                except Exception as e:
                    logger.error(f"[background_task] Failed to process URLs: {str(e)}", exc_info=True)
                    
                    # Update file status to FAILED
                    try:
                        db_error = SessionLocal()
                        try:
                            cls.update_file_status(
                                db=db_error, 
                                file_id=file_registry_id, 
                                status=FileStatus.FAILED
                            )
                        finally:
                            db_error.close()
                    except Exception:
                        logger.error(f"[background_task] Could not update file status to FAILED")
            
            # Add the task to background tasks
            background_tasks.add_task(
                process_urls_in_background, 
                urls, 
                plugin_name, 
                plugin_params, 
                collection_id, 
                file_registry.id
            )
            
            # Return immediate response with URL information
            return {
                "collection_id": collection_id,
                "collection_name": collection_name,
                "documents_added": 0,  # Initially 0 since processing will happen in background
                "success": True,
                "file_path": temp_file_path,
                "file_url": "",
                "original_filename": f"urls_{len(urls)}",
                "plugin_name": plugin_name,
                "file_registry_id": file_registry.id,
                "status": "processing"
            }
        except (ResourceNotFoundException, PluginNotFoundException, ValidationException):
            # Re-raise domain exceptions
            raise
        except Exception as e:
            logger.error(f"Failed to ingest URLs: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to ingest URLs: {str(e)}")
    
    @classmethod
    def ingest_file_to_collection(
        cls,
        collection_id: int,
        file: UploadFile,
        plugin_name: str,
        plugin_params: Dict[str, Any],
        db: Session,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Upload, process, and add a file to a collection in one operation.
        
        Args:
            collection_id: ID of the collection
            file: The file to upload and ingest
            plugin_name: Name of the ingestion plugin to use
            plugin_params: Parameters for the plugin
            db: Database session
            background_tasks: FastAPI background tasks
            
        Returns:
            Status information about the ingestion operation
            
        Raises:
            ResourceNotFoundException: If collection not found
            PluginNotFoundException: If plugin not found
            ProcessingException: If ingestion fails
        """
        # Verify collection exists in both SQLite and ChromaDB
        collection = cls.verify_collection_exists(collection_id)
        collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
        # Check if plugin exists
        plugin = cls.get_plugin(plugin_name)
        if not plugin:
            raise PluginNotFoundException(f"Ingestion plugin '{plugin_name}' not found")
        
        try:
            # Step 1: Upload file (this step remains synchronous)
            file_info = cls.save_uploaded_file(
                file=file,
                owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
                collection_name=collection_name
            )
            file_path = file_info["file_path"]
            file_url = file_info["file_url"]
            original_filename = file_info["original_filename"]
            owner = collection["owner"] if isinstance(collection, dict) else collection.owner
            
            # Step 2: Register the file in the FileRegistry with PROCESSING status
            file_registry = cls.register_file(
                db=db,
                collection_id=collection_id,
                file_path=file_path,
                file_url=file_url,
                original_filename=original_filename,
                plugin_name=plugin_name,
                plugin_params=plugin_params,
                owner=owner,
                document_count=0,  # Will be updated after processing
                content_type=file.content_type,
                status=FileStatus.PROCESSING  # Set initial status to PROCESSING
            )
            
            # Step 3: Schedule background task for processing and adding documents
            def process_file_in_background(file_path: str, plugin_name: str, params: dict, 
                                         collection_id: int, file_registry_id: int):
                try:
                    logger.debug(f"[background_task] Started processing file: {file_path}")
                    
                    # Create a new session for the background task
                    db_background = SessionLocal()
                    
                    try:
                        # Step 3.1: Process file with plugin
                        documents = cls.ingest_file(
                            file_path=file_path,
                            plugin_name=plugin_name,
                            plugin_params=params
                        )
                        
                        # Step 3.2: Add documents to collection
                        result = cls.add_documents_to_collection(
                            db=db_background,
                            collection_id=collection_id,
                            documents=documents
                        )
                        
                        # Step 3.3: Update file registry with completed status and document count
                        cls.update_file_status(
                            db=db_background, 
                            file_id=file_registry_id, 
                            status=FileStatus.COMPLETED
                        )
                        
                        # Update document count
                        file_reg = db_background.query(FileRegistry).filter(FileRegistry.id == file_registry_id).first()
                        if file_reg:
                            file_reg.document_count = len(documents)
                            db_background.commit()
                        
                        logger.debug(f"[background_task] Completed processing file {file_path} with {len(documents)} documents")
                    finally:
                        db_background.close()
                        
                except Exception as e:
                    logger.error(f"[background_task] Failed to process file {file_path}: {str(e)}", exc_info=True)
                    
                    # Update file status to FAILED
                    try:
                        db_error = SessionLocal()
                        try:
                            cls.update_file_status(
                                db=db_error, 
                                file_id=file_registry_id, 
                                status=FileStatus.FAILED
                            )
                        finally:
                            db_error.close()
                    except Exception:
                        logger.error(f"[background_task] Could not update file status to FAILED")
            
            # Add the task to background tasks
            background_tasks.add_task(
                process_file_in_background, 
                file_path, 
                plugin_name, 
                plugin_params, 
                collection_id, 
                file_registry.id
            )
            
            # Return immediate response with file information
            return {
                "collection_id": collection_id,
                "collection_name": collection_name,
                "documents_added": 0,  # Initially 0 since processing will happen in background
                "success": True,
                "file_path": file_path,
                "file_url": file_url,
                "original_filename": original_filename,
                "plugin_name": plugin_name,
                "file_registry_id": file_registry.id,
                "status": "processing"
            }
        except (ResourceNotFoundException, PluginNotFoundException):
            # Re-raise domain exceptions
            raise
        except Exception as e:
            logger.error(f"Failed to ingest file: {str(e)}", exc_info=True)
            raise ProcessingException(f"Failed to ingest file: {str(e)}")