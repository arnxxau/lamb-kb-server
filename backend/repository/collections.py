import os
import json
import logging
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
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
        db: Session = SessionLocal()
        
        if db.query(Collection).filter_by(name=name).first():
            db.close()
            logger.warning(f"Collection with name '{name}' already exists")
            return None

        params: Dict[str, Any] = {"name": name, "metadata": {"hnsw:space": "cosine"}}
        params["embedding_function"] = embedding_function

        client = get_chroma_client()
        chroma_collection = client.create_collection(**params)

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
        
        result = db_collection.to_dict()
        db.close()
        return result

    @staticmethod
    def get_collection(collection_id: int) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        collection = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            db.close()
            logger.info(f"Collection with ID {collection_id} not found")
            return None
        
        collection_dict = collection.to_dict()
        db.close()
        return collection_dict
    
    @staticmethod
    def _get_collection_by_name(name: str) -> Optional[Collection]:
        db = SessionLocal()
        collection = db.query(Collection).filter(Collection.name == name).first()
        db.close()
        return collection
    
    @staticmethod
    def list_collections(
        owner: Optional[str] = None,
        visibility: Optional[Visibility] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        db = SessionLocal()
        query = db.query(Collection)
        
        if owner:
            query = query.filter(Collection.owner == owner)
        if visibility:
            query = query.filter(Collection.visibility == visibility)
        
        collections = query.offset(skip).limit(limit).all()
        result = [col.to_dict() for col in collections]
        db.close()
        return result

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
        db: Session = SessionLocal()
        db_collection = db.query(Collection).get(collection_id)
        if not db_collection:
            db.close()
            return None

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

        if rename_from and rename_to:
            client = get_chroma_client()
            chroma_col = client.get_collection(rename_from)
            chroma_col.modify(name=rename_to)

        result = db_collection.to_dict()
        db.close()
        return result
    
    @staticmethod
    def delete_collection(collection_id: int) -> bool:
        db = SessionLocal()
        db_collection = db.query(Collection).filter(Collection.id == collection_id).first()
        
        if not db_collection:
            db.close()
            return False
        
        collection_name = db_collection.name
        chroma_client = get_chroma_client()
        
        # Attempt to delete from ChromaDB, log error if it fails but continue
        # with SQL deletion regardless
        result = chroma_client.delete_collection(collection_name)
        if not result:
            logger.error(f"Error deleting ChromaDB collection: {collection_name}")
        
        db.delete(db_collection)
        db.commit()
        db.close()
        return True
            
    @staticmethod
    def get_file_content(file_id: int) -> str:
        with SessionLocal() as db:
            registry = db.get(FileRegistry, file_id)
            if not registry:
                logger.error(f"File {file_id} not found")
                return ""

            collection_row = db.get(Collection, registry.collection_id)
            if not collection_row:
                logger.error(f"Collection {registry.collection_id} not found")
                return ""

        chroma = get_chroma_client()
        col = chroma.get_collection(collection_row.name)

        res = col.get(
            where={"file_id": file_id},
            include=["documents", "metadatas"]
        )
        if not res["documents"]:
            logger.error(f"No chunks stored for file {file_id}")
            return ""

        chunks = sorted(
            zip(res["documents"], res["metadatas"]),
            key=lambda p: p[1].get("chunk_index", 0)
        )
        return "\n".join(text for text, _ in chunks)