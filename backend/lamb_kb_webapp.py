#!/usr/bin/env python3
"""
Lamb Knowledge Base Web Application

A simple web app for interacting with the lamb-kb-server API. 
This app allows users to:
- View all collections for a given user
- View detailed information about a collection
- Query collections with custom parameters
- Debug ChromaDB collections
"""

import os
import json
import requests
import chromadb
import logging
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variable or use default
API_KEY = os.getenv("LAMB_API_KEY", "0p3n-w3bu!")
BASE_URL = os.getenv("LAMB_KB_SERVER_URL", "http://localhost:9090")

# Try to get ChromaDB path from environment or use a few common paths
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "/Users/ludo/Code/lamb-project/lamb-kb-server/backend/data/chromadb")
CHROMADB_PATHS = [
    CHROMADB_PATH,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/chromadb"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chromadb"), 
    os.path.join(os.path.abspath("."), "data/chromadb"),
    os.path.join(os.path.abspath(".."), "data/chromadb"),
    os.path.join(os.path.abspath("."), "backend/data/chromadb")
]

# Log all paths we're trying
logger.info(f"Working directory: {os.getcwd()}")
for path in CHROMADB_PATHS:
    if os.path.exists(path):
        logger.info(f"ChromaDB path exists: {path}")
    else:
        logger.warning(f"ChromaDB path does not exist: {path}")

app = Flask(__name__)
app.secret_key = os.urandom(24)

class LambKBClient:
    """Client for interacting with the lamb-kb-server API."""
    
    def __init__(self, base_url: str, api_key: str):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the lamb-kb-server API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the API."""
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", self.headers)
        
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {}
        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text[:500]}")
            raise
    
    def list_collections(self, owner=None) -> Dict[str, Any]:
        """List collections with optional owner filter."""
        params = {}
        if owner:
            params["owner"] = owner
        return self._request("get", "/collections", params=params)
    
    def get_collection(self, collection_id: int) -> Dict[str, Any]:
        """Get details of a specific collection."""
        return self._request("get", f"/collections/{collection_id}")
    
    def list_files(self, collection_id: int, status=None) -> List[Dict[str, Any]]:
        """List files in a collection with optional status filter."""
        params = {}
        if status:
            params["status"] = status
        return self._request("get", f"/collections/{collection_id}/files", params=params)
    
    def query_collection(self, collection_id: int, query_text: str, top_k: int = 5, 
                         threshold: float = 0.0, metadata_filter: Dict = None) -> Dict[str, Any]:
        """Query a collection for similar documents."""
        data = {
            "query_text": query_text,
            "top_k": top_k,
            "threshold": threshold,
            "plugin_params": {}
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            data["plugin_params"]["metadata_filter"] = metadata_filter
            data["plugin_params"]["include_metadata"] = True
            
        return self._request("post", f"/collections/{collection_id}/query", json=data)

# ChromaDB Helper Class
class ChromaDBHelper:
    """Helper class for directly interacting with ChromaDB."""
    
    def __init__(self, db_paths: List[str]):
        """Initialize the ChromaDB helper.
        
        Args:
            db_paths: Possible paths to the ChromaDB directory
        """
        self.db_paths = db_paths
        self.client = None
        self.db_path = None
        
        # Try to connect to ChromaDB using different paths
        for path in db_paths:
            try:
                logger.info(f"Trying to connect to ChromaDB at: {path}")
                if not os.path.exists(path):
                    logger.warning(f"Path does not exist: {path}")
                    continue
                    
                self.client = chromadb.PersistentClient(path=path)
                # Test connection by listing collections
                collections = self.client.list_collections()
                logger.info(f"Connected to ChromaDB at {path}, found {len(collections)} collections")
                self.db_path = path
                break
            except Exception as e:
                logger.error(f"Error connecting to ChromaDB at {path}: {e}")
        
        if self.client is None:
            logger.error("Failed to connect to ChromaDB with any of the provided paths")
    
    def get_collection_details(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection details
        """
        try:
            if self.client is None:
                return {
                    "name": collection_name,
                    "error": "Failed to connect to ChromaDB",
                    "success": False
                }
                
            collection = self.client.get_collection(collection_name)
            
            # Get basic collection info
            count = collection.count()
            
            # Get sample documents with metadata and embeddings
            sample_docs = collection.get(
                include=["embeddings", "documents", "metadatas"],
                limit=5
            )
            
            # Calculate embedding dimensions
            embedding_dims = [len(emb) for emb in sample_docs["embeddings"]] if sample_docs["embeddings"] else []
            
            # Extract unique chunking strategies from metadata
            chunking_strategies = {}
            metadata_keys = set()
            
            for meta in sample_docs["metadatas"]:
                if meta:
                    # Track all metadata keys
                    metadata_keys.update(meta.keys())
                    
                    # Track chunking info
                    if 'chunking_strategy' in meta:
                        strategy = meta['chunking_strategy']
                        if strategy not in chunking_strategies:
                            chunking_strategies[strategy] = 1
                        else:
                            chunking_strategies[strategy] += 1
                    
                    # Also track chunk units
                    if 'chunk_unit' in meta:
                        unit = meta['chunk_unit']
                        key = f"unit_{unit}"
                        if key not in chunking_strategies:
                            chunking_strategies[key] = 1
                        else:
                            chunking_strategies[key] += 1
            
            return {
                "name": collection_name,
                "document_count": count,
                "sample_metadata": sample_docs["metadatas"][:2] if sample_docs["metadatas"] else [],
                "embedding_dimensions": embedding_dims,
                "chunking_strategies": chunking_strategies,
                "metadata_keys": list(metadata_keys),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error getting collection details for {collection_name}: {e}")
            return {
                "name": collection_name,
                "error": str(e),
                "success": False
            }
    
    def list_collections(self) -> List[str]:
        """List all collections in ChromaDB."""
        try:
            if self.client is None:
                logger.error("Cannot list collections: ChromaDB client is None")
                return []
            
            try:
                collections = self.client.list_collections()
                # If collections is a list of strings, it's already what we want
                if collections and isinstance(collections[0], str):
                    logger.info(f"Found {len(collections)} collections in ChromaDB at {self.db_path}")
                    return collections
                
                # If collections is a list of objects, extract the names
                if hasattr(collections[0], 'name'):
                    collection_names = [col.name for col in collections]
                    logger.info(f"Found {len(collection_names)} collections in ChromaDB at {self.db_path}")
                    return collection_names
                
                # Fallback: something else is going on
                logger.warning(f"Unexpected collection format: {collections[0]}")
                return []
                
            except Exception as e:
                logger.error(f"Error listing ChromaDB collections: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error accessing ChromaDB: {e}")
            return []
            
    def get_document_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get document statistics from a collection."""
        try:
            if self.client is None:
                return {"error": "Failed to connect to ChromaDB"}
                
            collection = self.client.get_collection(collection_name)
            
            # Count by metadata values
            all_docs = collection.get(include=["metadatas"])
            
            # Track stats by different metadata fields
            stats = {
                "total_documents": len(all_docs["metadatas"]),
                "by_source": {},
                "by_chunk_unit": {},
                "by_chunking_strategy": {},
                "by_metadata_presence": {}
            }
            
            # Analyze metadata distribution
            for meta in all_docs["metadatas"]:
                if not meta:
                    continue
                    
                # Count by source
                if "source" in meta:
                    source = meta["source"]
                    stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
                
                # Count by chunk unit
                if "chunk_unit" in meta:
                    unit = meta["chunk_unit"]
                    stats["by_chunk_unit"][unit] = stats["by_chunk_unit"].get(unit, 0) + 1
                
                # Count by chunking strategy
                if "chunking_strategy" in meta:
                    strategy = meta["chunking_strategy"]
                    stats["by_chunking_strategy"][strategy] = stats["by_chunking_strategy"].get(strategy, 0) + 1
                
                # Count presence of each metadata field
                for key in meta:
                    stats["by_metadata_presence"][key] = stats["by_metadata_presence"].get(key, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}

    def get_advanced_diagnostics(self) -> Dict[str, Any]:
        """Perform advanced diagnostics on ChromaDB and SQLite.
        
        This method analyzes both the ChromaDB internal SQLite database and the 
        main application SQLite database to find inconsistencies and issues.
        
        Returns:
            Dictionary with diagnostic results
        """
        try:
            if self.client is None:
                return {
                    "success": False,
                    "error": "Failed to connect to ChromaDB",
                    "chromadb_path": self.db_path,
                }
            
            # Get ChromaDB collection names via API
            # These are just the raw collection names as strings
            collection_names = self.list_collections()
            
            # Convert to the same format as used in diagnostics
            chromadb_api_collections = [{"name": name, "count": 0} for name in collection_names]
            
            logger.info(f"API Collections: {collection_names}")
            
            # Connect to SQLite database
            sqlite_collections = self._get_sqlite_collections()
            logger.info(f"SQLite Collections: {[c['name'] for c in sqlite_collections]}")
            
            # Connect to ChromaDB SQLite database
            chromadb_internal_collections = self._get_chromadb_collections_from_sqlite()
            logger.info(f"ChromaDB Internal Collections: {[c['name'] for c in chromadb_internal_collections]}")
            
            # Examine UUID directories in ChromaDB
            uuid_dirs = self._examine_chromadb_directories()
            
            # Analyze segment info if available
            segment_info = self._analyze_segments()
            
            # Find mismatches
            mismatches = []
            
            # Check for collections in SQLite but not in ChromaDB
            for col in sqlite_collections:
                # We need to compare by name, not by ID
                col_name = col["name"].lower().strip()
                
                # Check in ChromaDB API collections
                found_in_api = any(name.lower().strip() == col_name for name in collection_names)
                
                # Check in ChromaDB internal collections
                found_in_internal = any(
                    internal_col["name"].lower().strip() == col_name 
                    for internal_col in chromadb_internal_collections
                )
                
                if not (found_in_api or found_in_internal):
                    mismatches.append({
                        "type": "missing_in_chromadb",
                        "name": col["name"],
                        "id": col["id"],
                        "severity": "high",
                        "message": f"Collection '{col['name']}' exists in SQLite but not in ChromaDB"
                    })
                elif not found_in_api and found_in_internal:
                    # Collection exists in internal ChromaDB but not in API
                    mismatches.append({
                        "type": "missing_in_chromadb_api",
                        "name": col["name"],
                        "id": col["id"],
                        "severity": "medium",
                        "message": f"Collection '{col['name']}' exists in SQLite and ChromaDB internal but not in ChromaDB API"
                    })
                elif found_in_api and not found_in_internal:
                    # Collection exists in API but not in internal ChromaDB
                    mismatches.append({
                        "type": "missing_in_chromadb_internal",
                        "name": col["name"],
                        "id": col["id"],
                        "severity": "medium",
                        "message": f"Collection '{col['name']}' exists in SQLite and ChromaDB API but not in internal ChromaDB"
                    })
            
            # Check for collections in ChromaDB but not in SQLite
            for name in collection_names:
                name_lower = name.lower().strip()
                found_in_sqlite = any(
                    sqlite_col["name"].lower().strip() == name_lower
                    for sqlite_col in sqlite_collections
                )
                
                if not found_in_sqlite:
                    mismatches.append({
                        "type": "missing_in_sqlite",
                        "name": name,
                        "severity": "medium",
                        "message": f"Collection '{name}' exists in ChromaDB but not in SQLite"
                    })
            
            # Check for UUID directories that don't match any collection
            for uuid_dir in uuid_dirs:
                uuid_value = uuid_dir['uuid']
                
                # Try to find this UUID in the ChromaDB internal collections
                matching_internal_collection = None
                for col in chromadb_internal_collections:
                    if col.get('id') == uuid_value:
                        matching_internal_collection = col
                        break
                
                if matching_internal_collection:
                    # UUID is associated with a collection in ChromaDB internal
                    # Check if it also exists in SQLite
                    collection_name = matching_internal_collection.get('name', '')
                    if collection_name:
                        matching_sqlite_collection = None
                        for col in sqlite_collections:
                            if col['name'].lower().strip() == collection_name.lower().strip():
                                matching_sqlite_collection = col
                                break
                        
                        if not matching_sqlite_collection:
                            # UUID exists in ChromaDB but not in SQLite
                            mismatches.append({
                                "type": "orphaned_uuid_not_in_sqlite",
                                "uuid": uuid_value,
                                "name": collection_name,
                                "files": uuid_dir["files"],
                                "severity": "medium",
                                "message": f"UUID directory '{uuid_value}' (collection '{collection_name}') exists in ChromaDB but not in SQLite"
                            })
                else:
                    # UUID doesn't match any known collection - truly orphaned
                    mismatches.append({
                        "type": "orphaned_uuid",
                        "uuid": uuid_value,
                        "files": uuid_dir["files"],
                        "severity": "medium",
                        "message": f"UUID directory '{uuid_value}' doesn't match any known ChromaDB collection"
                    })
            
            # Return all diagnostic data
            return {
                "success": True,
                "sqlite_collections": sqlite_collections,
                "chromadb_api_collections": chromadb_api_collections,
                "chromadb_internal_collections": chromadb_internal_collections,
                "uuid_directories": uuid_dirs,
                "segment_info": segment_info,
                "mismatches": mismatches,
                "total_mismatches": len(mismatches),
                "critical_mismatches": sum(1 for m in mismatches if m["severity"] == "high"),
                "medium_mismatches": sum(1 for m in mismatches if m["severity"] == "medium"),
                "minor_mismatches": sum(1 for m in mismatches if m["severity"] == "low")
            }
            
        except Exception as e:
            logger.error(f"Error performing advanced diagnostics: {e}")
            return {
                "success": False,
                "error": str(e),
                "chromadb_path": self.db_path
            }
    
    def _get_sqlite_collections(self) -> List[Dict[str, Any]]:
        """Get collections from the SQLite database"""
        # Define SQLite path
        sqlite_path = os.path.join(os.path.dirname(self.db_path), "lamb-kb-server.db")
        if not os.path.exists(sqlite_path):
            logger.error(f"SQLite database not found at: {sqlite_path}")
            return []
        
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, owner, creation_date, embeddings_model FROM collections")
        collections = [dict(row) for row in cursor.fetchall()]
        
        for collection in collections:
            # Format embeddings model if it's a JSON string
            if isinstance(collection['embeddings_model'], str):
                try:
                    collection['embeddings_model'] = json.loads(collection['embeddings_model'])
                except json.JSONDecodeError:
                    pass
        
        conn.close()
        return collections
    
    def _get_chromadb_collections_from_sqlite(self) -> List[Dict[str, Any]]:
        """Get collections directly from ChromaDB SQLite"""
        chroma_db_path = os.path.join(self.db_path, 'chroma.sqlite3')
        if not os.path.exists(chroma_db_path):
            logger.error(f"ChromaDB SQLite file not found at: {chroma_db_path}")
            return []
        
        conn = sqlite3.connect(chroma_db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        
        # First check the actual columns in the collections table
        try:
            cursor.execute("PRAGMA table_info(collections)")
            available_columns = [row[1] for row in cursor.fetchall()]
            
            if not available_columns:
                logger.warning("No columns found in collections table")
                conn.close()
                return []
            
            # Log available columns
            logger.info(f"Available columns in collections table: {available_columns}")
            
            # Construct a query based on available columns
            # In ChromaDB, the 'id' field is the UUID, not a simple integer ID
            query_columns = []
            if 'id' in available_columns:
                query_columns.append('id')
            if 'name' in available_columns:
                query_columns.append('name')
            
            if not query_columns:
                logger.warning("Required columns 'id' and 'name' not found in collections table")
                conn.close()
                return []
            
            # Execute the query to get collections
            query = f"SELECT {', '.join(query_columns)} FROM collections"
            cursor.execute(query)
            
            collections = []
            for row in cursor.fetchall():
                collection_data = {}
                for i, col_name in enumerate(query_columns):
                    collection_data[col_name] = row[i]
                
                # Validate UUID format if present
                if 'id' in collection_data:
                    try:
                        # Try to parse as UUID to validate
                        uuid_obj = uuid.UUID(collection_data['id'])
                        # It's a valid UUID
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid UUID format for collection {collection_data.get('name', 'unknown')}: {collection_data['id']}")
                
                collections.append(collection_data)
            
            # Log the results
            logger.info(f"Found {len(collections)} collections in ChromaDB SQLite")
            for col in collections:
                logger.info(f"Collection from ChromaDB SQLite: name='{col.get('name', '')}', id='{col.get('id', '')}'")
            
            # Get additional metadata if available
            if 'collection_metadata' in self._get_table_names(conn):
                for collection in collections:
                    if 'id' not in collection:
                        continue
                    
                    try:
                        metadata_cursor = conn.cursor()
                        # Check columns in collection_metadata table first
                        metadata_cursor.execute("PRAGMA table_info(collection_metadata)")
                        metadata_columns = [row[1] for row in metadata_cursor.fetchall()]
                        
                        # Adjust query based on available columns
                        if 'key' in metadata_columns and 'value' in metadata_columns:
                            metadata_cursor.execute("SELECT key, value FROM collection_metadata WHERE collection_id = ?", 
                                                  (collection['id'],))
                            metadata = {row[0]: row[1] for row in metadata_cursor.fetchall()}
                        elif 'key' in metadata_columns and 'str_value' in metadata_columns:
                            # Some versions use 'str_value' instead of 'value'
                            metadata_cursor.execute("SELECT key, str_value FROM collection_metadata WHERE collection_id = ?", 
                                                  (collection['id'],))
                            metadata = {row[0]: row[1] for row in metadata_cursor.fetchall()}
                        else:
                            logger.warning(f"Cannot fetch metadata - columns missing. Available: {metadata_columns}")
                            metadata = {}
                            
                        collection['metadata'] = metadata
                    except Exception as e:
                        logger.error(f"Error fetching metadata for collection {collection.get('id', 'unknown')}: {e}")
            
            conn.close()
            return collections
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB SQLite: {e}")
            conn.close()
            return []
    
    def _get_table_names(self, conn) -> List[str]:
        """Get all table names from a SQLite connection"""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def _examine_chromadb_directories(self) -> List[Dict[str, Any]]:
        """Examine UUIDs in the ChromaDB directory"""
        if not self.db_path or not os.path.exists(self.db_path):
            logger.error(f"ChromaDB directory not found at: {self.db_path}")
            return []
        
        # Get all directory entries that look like UUIDs
        contents = os.listdir(self.db_path)
        uuid_dirs = []
        
        for item in contents:
            # Check if it's a directory and looks like a UUID
            item_path = os.path.join(self.db_path, item)
            if os.path.isdir(item_path):
                try:
                    # Try to parse as UUID to validate
                    uuid_obj = uuid.UUID(item)
                    uuid_dirs.append({
                        'uuid': item,
                        'path': item_path,
                        'files': os.listdir(item_path)
                    })
                except (ValueError, TypeError):
                    # Not a UUID, ignore
                    pass
                    
        return uuid_dirs
    
    def _analyze_segments(self) -> Dict[str, Any]:
        """Analyze ChromaDB segments"""
        chroma_db_path = os.path.join(self.db_path, 'chroma.sqlite3')
        if not os.path.exists(chroma_db_path):
            logger.error(f"ChromaDB SQLite file not found at: {chroma_db_path}")
            return {}
        
        conn = sqlite3.connect(chroma_db_path)
        cursor = conn.cursor()
        
        tables = self._get_table_names(conn)
        
        if 'segments' not in tables or 'embeddings' not in tables:
            conn.close()
            return {"error": "Segments or embeddings table not found"}
        
        # Check segments table structure
        cursor.execute("PRAGMA table_info(segments)")
        segment_columns = [row[1] for row in cursor.fetchall()]
        
        # Check embeddings table structure
        cursor.execute("PRAGMA table_info(embeddings)")
        embedding_columns = [row[1] for row in cursor.fetchall()]
        
        # Get segment data
        segments_data = []
        
        if 'id' in segment_columns and 'collection' in segment_columns:
            cursor.execute("SELECT id, collection FROM segments")
            for row in cursor.fetchall():
                segment_id, collection_id = row
                
                # Count embeddings in this segment
                if 'segment_id' in embedding_columns:
                    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE segment_id = ?", (segment_id,))
                    embedding_count = cursor.fetchone()[0]
                else:
                    embedding_count = "unknown"
                
                segments_data.append({
                    "id": segment_id,
                    "collection_id": collection_id,
                    "embedding_count": embedding_count
                })
        
        conn.close()
        
        return {
            "segment_columns": segment_columns,
            "embedding_columns": embedding_columns,
            "segments": segments_data
        }

# Create client instances
client = LambKBClient(BASE_URL, API_KEY)
chroma_helper = ChromaDBHelper(CHROMADB_PATHS)

@app.route('/')
def index():
    """Home page - show a form to search for collections by owner."""
    return render_template('index.html')

@app.route('/collections', methods=['GET'])
def list_collections():
    """List collections, optionally filtered by owner."""
    owner = request.args.get('owner', '')
    try:
        if owner:
            collections = client.list_collections(owner=owner)
        else:
            collections = client.list_collections()
        return render_template('collections.html', collections=collections.get('items', []), owner=owner)
    except Exception as e:
        flash(f"Error fetching collections: {str(e)}", "error")
        return render_template('collections.html', collections=[], owner=owner)

@app.route('/collections/<int:collection_id>')
def view_collection(collection_id):
    """View detailed information about a collection."""
    try:
        collection = client.get_collection(collection_id)
        files = client.list_files(collection_id)
        
        # Calculate some statistics
        total_documents = sum(file.get('document_count', 0) for file in files)
        file_count = len(files)
        
        return render_template(
            'collection_details.html', 
            collection=collection, 
            files=files, 
            file_count=file_count,
            total_documents=total_documents
        )
    except Exception as e:
        flash(f"Error fetching collection details: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/collections/<int:collection_id>/query', methods=['GET', 'POST'])
def query_collection(collection_id):
    """Query a collection and display results."""
    try:
        collection = client.get_collection(collection_id)
        
        if request.method == 'POST':
            query_text = request.form.get('query_text', '')
            top_k = int(request.form.get('top_k', 5))
            threshold = float(request.form.get('threshold', 0.0))
            include_all_metadata = request.form.get('include_all_metadata') == 'on'
            
            if not query_text:
                flash("Please enter a query text", "error")
                return render_template('query.html', collection=collection)
            
            metadata_filter = {} if include_all_metadata else None
            results = client.query_collection(
                collection_id, 
                query_text, 
                top_k, 
                threshold, 
                metadata_filter
            )
            
            return render_template(
                'query_results.html', 
                collection=collection, 
                query_text=query_text,
                top_k=top_k,
                threshold=threshold,
                results=results,
                include_all_metadata=include_all_metadata
            )
        
        return render_template('query.html', collection=collection)
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('view_collection', collection_id=collection_id))

@app.route('/debug/chromadb')
def debug_chromadb():
    """Debug ChromaDB collections."""
    try:
        # Get path info for debugging
        paths_info = []
        for path in CHROMADB_PATHS:
            exists = os.path.exists(path)
            paths_info.append({
                "path": path,
                "exists": exists,
                "is_current": path == chroma_helper.db_path
            })
        
        # Get collections from different sources
        api_collections = chroma_helper.list_collections()
        logger.info(f"Raw API Collections: {api_collections}")
        
        # Get diagnostics data
        diagnostics = chroma_helper.get_advanced_diagnostics()
        
        sqlite_collections = diagnostics.get('sqlite_collections', [])
        chromadb_internal_collections = diagnostics.get('chromadb_internal_collections', [])
        
        # Create explicit mapping for template
        collection_mapping = []
        
        # First add SQLite collections
        for col in sqlite_collections:
            col_name_lower = col['name'].lower().strip()
            
            # Find in API collections
            found_in_api = False
            api_name = None
            for api_col in api_collections:
                if col_name_lower == api_col.lower().strip():
                    found_in_api = True
                    api_name = api_col
                    break
            
            # Find in internal collections
            found_uuid = None
            for internal_col in chromadb_internal_collections:
                if col_name_lower == internal_col['name'].lower().strip():
                    found_uuid = internal_col['id']
                    break
            
            collection_mapping.append({
                'sqlite_name': col['name'],
                'sqlite_id': col['id'],
                'chroma_name': api_name if found_in_api else None,
                'uuid': found_uuid,
                'found_in_api': found_in_api,
                'found_uuid': found_uuid is not None
            })
        
        # Then add ChromaDB collections not in SQLite
        for api_col in api_collections:
            api_name_lower = api_col.lower().strip()
            
            # Check if this API collection is already in the mapping
            already_mapped = False
            for mapping in collection_mapping:
                if mapping['chroma_name'] and mapping['chroma_name'].lower().strip() == api_name_lower:
                    already_mapped = True
                    break
            
            if not already_mapped:
                # Find in internal collections
                found_uuid = None
                for internal_col in chromadb_internal_collections:
                    if api_name_lower == internal_col['name'].lower().strip():
                        found_uuid = internal_col['id']
                        break
                
                mapping = {
                    'sqlite_name': None,
                    'sqlite_id': None, 
                    'chroma_name': api_col,
                    'uuid': found_uuid,
                    'found_in_api': True,
                    'found_uuid': found_uuid is not None,
                    'only_in_chroma': True
                }
                collection_mapping.append(mapping)
        
        # Print some debug info
        logger.info(f"Collection Mapping: {collection_mapping}")
        
        return render_template('debug_chromadb.html', 
                               collections=api_collections,
                               sqlite_collections=sqlite_collections,
                               chromadb_internal_collections=chromadb_internal_collections,
                               collection_mapping=collection_mapping,
                               db_path=chroma_helper.db_path,
                               paths_info=paths_info)
    except Exception as e:
        logger.error(f"Error accessing ChromaDB: {str(e)}")
        return redirect(url_for('index'))

@app.route('/debug/chromadb/<collection_name>')
def debug_collection(collection_name):
    """Debug a specific ChromaDB collection."""
    try:
        collection_details = chroma_helper.get_collection_details(collection_name)
        document_stats = chroma_helper.get_document_stats(collection_name)
        
        # Get all KB collections to map ID to name
        kb_collections = client.list_collections().get('items', [])
        kb_collection_map = {col['name']: col for col in kb_collections}
        
        # Find matching KB collection
        kb_collection = kb_collection_map.get(collection_name, {})
        
        return render_template(
            'debug_collection.html',
            collection_details=collection_details,
            document_stats=document_stats,
            kb_collection=kb_collection
        )
    except Exception as e:
        flash(f"Error inspecting ChromaDB collection: {str(e)}", "error")
        return redirect(url_for('debug_chromadb'))

@app.route('/api/collections')
def api_list_collections():
    """API endpoint to list collections."""
    owner = request.args.get('owner', '')
    try:
        if owner:
            collections = client.list_collections(owner=owner)
        else:
            collections = client.list_collections()
        return jsonify(collections)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/collections/<int:collection_id>')
def api_get_collection(collection_id):
    """API endpoint to get collection details."""
    try:
        collection = client.get_collection(collection_id)
        files = client.list_files(collection_id)
        
        # Add calculated statistics
        collection['file_count'] = len(files)
        collection['total_documents'] = sum(file.get('document_count', 0) for file in files)
        collection['files'] = files
        
        return jsonify(collection)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/collections/<int:collection_id>/query', methods=['POST'])
def api_query_collection(collection_id):
    """API endpoint to query a collection."""
    try:
        data = request.get_json()
        query_text = data.get('query_text', '')
        top_k = int(data.get('top_k', 5))
        threshold = float(data.get('threshold', 0.0))
        metadata_filter = data.get('metadata_filter')
        
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        results = client.query_collection(collection_id, query_text, top_k, threshold, metadata_filter)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/chromadb')
def api_list_chromadb_collections():
    """API endpoint to list ChromaDB collections."""
    try:
        collections = chroma_helper.list_collections()
        return jsonify({"collections": collections, "db_path": chroma_helper.db_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/chromadb/<collection_name>')
def api_debug_collection(collection_name):
    """API endpoint to debug a ChromaDB collection."""
    try:
        collection_details = chroma_helper.get_collection_details(collection_name)
        document_stats = chroma_helper.get_document_stats(collection_name)
        
        return jsonify({
            "collection_details": collection_details,
            "document_stats": document_stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/diagnostics')
def advanced_diagnostics():
    """Display advanced diagnostic information about ChromaDB and SQLite."""
    try:
        # Run diagnostics to get basic information
        diagnostics_raw = chroma_helper.get_advanced_diagnostics()
        
        # Get raw collections from different sources
        api_collections = chroma_helper.list_collections()
        logger.info(f"Advanced Diagnostics Raw API Collections: {api_collections}")
        
        sqlite_collections = diagnostics_raw.get('sqlite_collections', [])
        chromadb_internal_collections = diagnostics_raw.get('chromadb_internal_collections', [])
        
        # Print detailed debug info about collections
        logger.info(f"API Collections {len(api_collections)}: {api_collections}")
        logger.info(f"SQLite Collections {len(sqlite_collections)}: {[c['name'] for c in sqlite_collections]}")
        logger.info(f"ChromaDB Internal Collections {len(chromadb_internal_collections)}: {[(c.get('name', ''), c.get('id', '')) for c in chromadb_internal_collections]}")
        
        # Create proper collection mappings (same logic as debug_chromadb)
        collection_mapping = []
        mismatches = []
        
        # First add SQLite collections
        for col in sqlite_collections:
            col_name_lower = col['name'].lower().strip()
            
            # Find in API collections
            found_in_api = False
            api_name = None
            for api_col in api_collections:
                if col_name_lower == api_col.lower().strip():
                    found_in_api = True
                    api_name = api_col
                    break
            
            # Find in internal collections
            found_uuid = None
            matching_internal = None
            for internal_col in chromadb_internal_collections:
                internal_name = internal_col.get('name', '').lower().strip()
                if col_name_lower == internal_name:
                    found_uuid = internal_col.get('id', '')
                    matching_internal = internal_col
                    break
            
            # Log detailed debug info for this mapping
            logger.info(f"SQLite collection '{col['name']}': API match: {api_name}, UUID: {found_uuid}")
            
            mapping = {
                'sqlite_name': col['name'],
                'sqlite_id': col['id'],
                'chroma_name': api_name if found_in_api else None,
                'uuid': found_uuid,
                'found_in_api': found_in_api,
                'found_uuid': found_uuid is not None and found_uuid != ''
            }
            collection_mapping.append(mapping)
            
            # Add mismatches if necessary
            if not (found_in_api or found_uuid):
                mismatches.append({
                    "type": "missing_in_chromadb",
                    "name": col["name"],
                    "id": col["id"],
                    "severity": "high",
                    "message": f"Collection '{col['name']}' exists in SQLite but not in ChromaDB"
                })
            elif not found_in_api and found_uuid:
                mismatches.append({
                    "type": "missing_in_chromadb_api",
                    "name": col["name"],
                    "id": col["id"],
                    "severity": "medium",
                    "message": f"Collection '{col['name']}' exists in SQLite and ChromaDB internal but not in ChromaDB API"
                })
            elif found_in_api and not found_uuid:
                mismatches.append({
                    "type": "missing_in_chromadb_internal",
                    "name": col["name"],
                    "id": col["id"],
                    "severity": "medium",
                    "message": f"Collection '{col['name']}' exists in SQLite and ChromaDB API but not in internal ChromaDB"
                })
        
        # Then add ChromaDB collections not in SQLite
        for api_col in api_collections:
            api_name_lower = api_col.lower().strip()
            
            # Check if this API collection is already in the mapping
            already_mapped = False
            for mapping in collection_mapping:
                if mapping['chroma_name'] and mapping['chroma_name'].lower().strip() == api_name_lower:
                    already_mapped = True
                    break
            
            if not already_mapped:
                # Find in internal collections
                found_uuid = None
                for internal_col in chromadb_internal_collections:
                    internal_name = internal_col.get('name', '').lower().strip() 
                    if api_name_lower == internal_name:
                        found_uuid = internal_col.get('id', '')
                        break
                
                mapping = {
                    'sqlite_name': None,
                    'sqlite_id': None, 
                    'chroma_name': api_col,
                    'uuid': found_uuid,
                    'found_in_api': True,
                    'found_uuid': found_uuid is not None and found_uuid != '',
                    'only_in_chroma': True
                }
                collection_mapping.append(mapping)
                
                # Add mismatch
                mismatches.append({
                    "type": "missing_in_sqlite",
                    "name": api_col,
                    "severity": "medium",
                    "message": f"Collection '{api_col}' exists in ChromaDB but not in SQLite"
                })
        
        # Replace mismatches in diagnostics with our corrected version
        diagnostics = diagnostics_raw.copy()
        diagnostics['mismatches'] = mismatches
        diagnostics['collection_mapping'] = collection_mapping
        diagnostics['total_mismatches'] = len(mismatches)
        diagnostics['critical_mismatches'] = sum(1 for m in mismatches if m["severity"] == "high")
        diagnostics['medium_mismatches'] = sum(1 for m in mismatches if m["severity"] == "medium")
        diagnostics['minor_mismatches'] = sum(1 for m in mismatches if m["severity"] == "low")
        
        return render_template(
            'advanced_diagnostics.html',
            diagnostics=diagnostics,
            db_path=chroma_helper.db_path
        )
    except Exception as e:
        flash(f"Error running diagnostics: {str(e)}", "error")
        return redirect(url_for('debug_chromadb'))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Log ChromaDB connection status
    if chroma_helper.client:
        logger.info(f"Successfully connected to ChromaDB at: {chroma_helper.db_path}")
        collections = chroma_helper.list_collections()
        logger.info(f"Found {len(collections)} collections: {', '.join(collections)}")
    else:
        logger.error("Failed to connect to ChromaDB. Debug paths:")
        for path in CHROMADB_PATHS:
            if os.path.exists(path):
                logger.info(f"  Path exists: {path}")
                try:
                    files = os.listdir(path)
                    logger.info(f"  Contents: {files[:10]}")
                except Exception as e:
                    logger.error(f"  Error listing contents: {e}")
            else:
                logger.warning(f"  Path does not exist: {path}")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=8080, debug=True) 