import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from .models import FileRegistry, FileStatus
from .connection import get_db, get_chroma_client, get_embedding_function, SessionLocal
from .collections import CollectionRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionRepository:
    @staticmethod
    def register_file(
        collection_id: int, 
        file_path: str, 
        file_url: str, 
        original_filename: str, 
        plugin_name: str, 
        plugin_params: Dict[str, Any],
        owner: str,
        document_count: int = 0,
        content_type: Optional[str] = None,
        status: FileStatus = FileStatus.COMPLETED
    ) -> Dict[str, Any]:
        db = SessionLocal()
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        if isinstance(plugin_params, str):
            plugin_params = json.loads(plugin_params) if plugin_params.strip() else {}
                
        file_registry = FileRegistry(
            collection_id=collection_id,
            original_filename=original_filename,
            file_path=file_path,
            file_url=file_url,
            file_size=file_size,
            content_type=content_type,
            plugin_name=plugin_name,
            plugin_params=plugin_params,
            status=status,
            document_count=document_count,
            owner=owner
        )
        
        db.add(file_registry)
        db.commit()
        db.refresh(file_registry)
        
        result = {
            "id": file_registry.id,
            "collection_id": file_registry.collection_id,
            "file_path": file_registry.file_path,
            "file_url": file_registry.file_url,
            "original_filename": file_registry.original_filename,
            "status": file_registry.status.value,
            "created_at": file_registry.created_at.isoformat() if file_registry.created_at else None
        }
        db.close()
        return result
    
    @staticmethod
    def update_file_status(file_id: int, status: FileStatus) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if file_registry:
            file_registry.status = status
            file_registry.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(file_registry)
            
            result = {
                "id": file_registry.id,
                "status": file_registry.status.value,
                "updated_at": file_registry.updated_at.isoformat() if file_registry.updated_at else None
            }
            db.close()
            return result
        db.close()
        return None
    
    @staticmethod
    def update_document_count(file_id: int, document_count: int) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if file_registry:
            file_registry.document_count = document_count
            file_registry.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(file_registry)
            
            result = {
                "id": file_registry.id,
                "document_count": file_registry.document_count,
                "updated_at": file_registry.updated_at.isoformat() if file_registry.updated_at else None
            }
            db.close()
            return result
        db.close()
        return None
    
    @staticmethod
    def add_documents_to_collection(
        collection_id: int,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        db = SessionLocal()
        db_collection = CollectionRepository.get_collection(collection_id)
        if not db_collection:
            logger.error(f"Collection not found")
            db.close()
            return None
        
        collection_name = db_collection['name'] if isinstance(db_collection, dict) else db_collection.name
        embedding_config = db_collection.embeddings_model if not isinstance(db_collection, dict) else db_collection['embeddings_model']
        vendor = embedding_config.get("vendor", "")
        model_name = embedding_config.get("model", "")
        
        chroma_client = get_chroma_client()
        
        collection_embedding_function = get_embedding_function(db_collection)
        if not collection_embedding_function:
            logger.error("Failed to create embedding function")
            db.close()
            return None
            
        test_result = collection_embedding_function(["Test embedding function verification"])
        
        collections = chroma_client.list_collections()
        
        collection_exists = False
        if collections and isinstance(collections[0], str):
            collection_exists = collection_name in collections
        else:
            collection_exists = any(col.name == collection_name for col in collections)
            if not collection_exists:
                chroma_client.get_collection(name=collection_name)
                collection_exists = True
        
        if not collection_exists:
            logger.error(f"ChromaDB collection does not exist: {collection_name}")
            db.close()
            return None
        
        chroma_collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=collection_embedding_function
        )
        
        existing_metadata = chroma_collection.metadata
        if existing_metadata:
            existing_embeddings_model = existing_metadata.get("embeddings_model", "{}")
            
            existing_config = {}
            if isinstance(existing_embeddings_model, str):
                if existing_embeddings_model.strip() and existing_embeddings_model != "{}":
                    existing_config = json.loads(existing_embeddings_model)
            else:
                existing_config = existing_embeddings_model
                    
            existing_vendor = existing_config.get("vendor", "")
            existing_model = existing_config.get("model", "")
            
            if existing_vendor != vendor or existing_model != model_name:
                logger.warning(f"Embedding model mismatch detected! Using the embedding function from SQLite record")
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = f"{uuid.uuid4().hex}"
            ids.append(doc_id)
            texts.append(doc["text"])
            
            metadata = doc.get("metadata", {}).copy()
            metadata["document_id"] = doc_id
            metadata["ingestion_timestamp"] = datetime.utcnow().isoformat()
            metadata["embedding_vendor"] = vendor
            metadata["embedding_model"] = model_name
            
            metadatas.append(metadata)
        
        start_time = time.time()
        
        batch_size = 5
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            batch_start_time = time.time()
            
            chroma_collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            
            batch_end_time = time.time()
            logger.debug(f"Batch {i//batch_size + 1} completed in {batch_end_time - batch_start_time:.2f} seconds")
        
        end_time = time.time()
        logger.debug(f"ChromaDB add operation completed in {end_time - start_time:.2f} seconds")
        
        result = {
            "collection_id": collection_id,
            "collection_name": collection_name,
            "documents_added": len(documents),
            "success": True,
            "embedding_info": {
                "vendor": vendor,
                "model": model_name
            }
        }
        db.close()
        return result
    
    @staticmethod
    def process_file(file_path: str, plugin_name: str, params: dict, 
                     collection_id: int, file_registry_id: int, documents: List[Dict[str, Any]]):
        result = IngestionRepository.add_documents_to_collection(
            collection_id=collection_id,
            documents=documents
        )
        
        result = IngestionRepository.update_file_status(
            file_id=file_registry_id, 
            status=FileStatus.COMPLETED
        )
        if not result:
            logger.error(f"File with ID {file_registry_id} not found")
        
        IngestionRepository.update_document_count(
            file_id=file_registry_id,
            document_count=len(documents)
        )
        
    @staticmethod
    def process_urls(temp_file_path: str, plugin_name: str, params: dict, 
                     collection_id: int, file_registry_id: int, documents: List[Dict[str, Any]]):
        result = IngestionRepository.add_documents_to_collection(
            collection_id=collection_id,
            documents=documents
        )
        
        result = IngestionRepository.update_file_status(
            file_id=file_registry_id, 
            status=FileStatus.COMPLETED
        )
        if not result:
            logger.error(f"File with ID {file_registry_id} not found")
        
        IngestionRepository.update_document_count(
            file_id=file_registry_id,
            document_count=len(documents)
        )