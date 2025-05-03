import os
import json
import logging
import tempfile
import uuid
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, status, Query, File, Form, UploadFile, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Depends
from pydantic import BaseModel, Field

from repository.models import Collection, Visibility, FileRegistry, FileStatus
from repository.ingestion import IngestionRepository
from services.collections import CollectionsService
from services.ingestion import IngestionService
from services.query import QueryService

from schemas.collection import (
    CollectionCreate, 
    CollectionUpdate, 
    CollectionResponse, 
    CollectionList,
    EmbeddingsModel
)

from exceptions import (
    DomainException,
    ResourceNotFoundException,
    ValidationException,
    ResourceAlreadyExistsException,
    ConfigurationException,
    ProcessingException,
    AuthenticationException,
    AuthorizationException,
    DatabaseException,
    PluginNotFoundException,
    ExternalServiceException,
    FileNotFoundException
)

from schemas.ingestion import (
    IngestionPluginInfo,
    IngestFileRequest,
    IngestFileResponse,
    IngestURLRequest,
    IngestURLResponse,
    AddDocumentsRequest,
    AddDocumentsResponse,
    PreviewURLRequest, 
    PreviewURLResponse
)

from schemas.query import (
    QueryRequest,
    QueryResponse,
    QueryPluginInfo
)

logger = logging.getLogger(__name__)

class FileRegistryResponse(BaseModel):
    id: int = Field(..., description="ID of the file registry entry")
    collection_id: int = Field(..., description="ID of the collection")
    original_filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to the file on the server")
    file_url: str = Field(..., description="URL to access the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    plugin_name: str = Field(..., description="Name of the ingestion plugin used")
    plugin_params: Dict[str, Any] = Field(..., description="Parameters used for ingestion")
    status: str = Field(..., description="Status of the file")
    document_count: int = Field(..., description="Number of documents created from this file")
    created_at: str = Field(..., description="Timestamp when the file was added")
    updated_at: str = Field(..., description="Timestamp when the file record was last updated")
    owner: str = Field(..., description="Owner of the file")

API_KEY = os.getenv("LAMB_API_KEY", "0p3n-w3bu!")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the server", example="ok")
    version: str = Field(..., description="Server version", example="0.1.0")

class MessageResponse(BaseModel):
    message: str = Field(..., description="Response message", example="Hello World from the Lamb Knowledge Base Server!")

class DatabaseStatusResponse(BaseModel):
    status: str = Field(..., description="Status of the server", example="In-memory mode")
    info: Dict[str, Any] = Field(..., description="Information about the in-memory storage")

app = FastAPI(
    title="Lamb Knowledge Base Server",
    description="A dedicated knowledge base server designed to provide robust vector functionality for the LAMB project and to serve as a Model Context Protocol (MCP) server.",
    version="0.1.0",
    contact={
        "name": "LAMB Project Team",
    },
    license_info={
        "name": "GNU General Public License v3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.en.html"
    },
)

@app.exception_handler(ResourceNotFoundException)
async def resource_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )

@app.exception_handler(ValidationException)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )

@app.exception_handler(ResourceAlreadyExistsException)
async def resource_already_exists_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc)},
    )

@app.exception_handler(ConfigurationException)
async def configuration_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )

@app.exception_handler(ProcessingException)
async def processing_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

@app.exception_handler(AuthenticationException)
async def authentication_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": str(exc)},
    )

@app.exception_handler(AuthorizationException)
async def authorization_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(exc)},
    )

@app.exception_handler(DatabaseException)
async def database_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

@app.exception_handler(PluginNotFoundException)
async def plugin_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )

@app.exception_handler(FileNotFoundException)
async def file_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )

@app.exception_handler(DomainException)
async def domain_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

security = HTTPBearer(
    scheme_name="Bearer Authentication",
    description="Enter the API token as a Bearer token",
    auto_error=True,
)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.on_event("startup")
async def startup_event():
    from plugins.base import discover_plugins
    discover_plugins("plugins")
    IngestionService._ensure_dirs()

static_dir = IngestionService.STATIC_DIR
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/query/plugins",
    response_model=List[QueryPluginInfo],
    summary="List available query plugins",
    description="Get a list of available query plugins.",
    tags=["Query"]
)
async def list_query_plugins(token: str = Depends(verify_token)):
    return QueryService.list_plugins()

@app.post(
    "/collections/{collection_id}/query",
    response_model=QueryResponse,
    summary="Query a collection",
    description="Query a collection using a specified plugin.",
    tags=["Query"]
)
async def query_collection(
    collection_id: int,
    request: QueryRequest,
    plugin_name: str = Query("simple_query", description="Name of the query plugin to use"),
    token: str = Depends(verify_token)
):
    plugin_params = request.plugin_params or {}
    
    if "top_k" not in plugin_params and request.top_k is not None:
        plugin_params["top_k"] = request.top_k
    if "threshold" not in plugin_params and request.threshold is not None:
        plugin_params["threshold"] = request.threshold
    
    return QueryService.query_collection(
        collection_id=collection_id,
        query_text=request.query_text,
        plugin_name=plugin_name,
        plugin_params=plugin_params
    )

@app.get(
    "/", 
    response_model=MessageResponse,
    summary="Root endpoint",
    description="Returns a welcome message to confirm the server is running.",
    tags=["System"]
)
async def root(token: str = Depends(verify_token)):
    return {"message": "Hello World from the Lamb Knowledge Base Server!"}

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the server.",
    tags=["System"]
)
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.get(
    "/ingestion/plugins",
    response_model=List[IngestionPluginInfo],
    summary="List ingestion plugins",
    description="List all available document ingestion plugins.",
    tags=["Ingestion"]
)
async def list_ingestion_plugins(token: str = Depends(verify_token)):
    return IngestionService.list_plugins()

@app.post(
    "/collections/{collection_id}/ingest-url",
    response_model=AddDocumentsResponse,
    summary="Ingest content from URLs directly into a collection",
    description="Fetch, process, and add content from URLs to a collection in one operation.",
    tags=["Ingestion"]
)
async def ingest_url_to_collection(
    collection_id: int,
    request: IngestURLRequest,
    token: str = Depends(verify_token),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not request.urls:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No URLs provided")
    
    collection = CollectionsService.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection with ID {collection_id} not found")
    
    plugin = IngestionService.get_plugin(request.plugin_name)
    if not plugin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plugin {request.plugin_name} not found")
    
    temp_dir = os.path.join(tempfile.gettempdir(), "url_ingestion")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.url")
    
    with open(temp_file_path, "w") as f:
        for url in request.urls:
            f.write(f"{url}\n")
    
    first_url = request.urls[0]
    collection_name = collection.name if hasattr(collection, 'name') else collection['name']
    owner = collection.owner if hasattr(collection, 'owner') else collection['owner']
    
    file_registry = IngestionRepository.register_file(
        collection_id=collection_id,
        file_path=temp_file_path,
        file_url=first_url,
        original_filename=first_url,
        plugin_name="url_ingest",
        plugin_params={"urls": request.urls, **request.plugin_params},
        owner=owner,
        document_count=0,
        content_type="text/plain",
        status=FileStatus.PROCESSING
    )
    
    if not file_registry:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register URL ingestion")
    
    background_tasks.add_task(
        IngestionService.process_urls,
        urls=request.urls,
        plugin_name=request.plugin_name,
        plugin_params=request.plugin_params,
        collection_id=collection_id,
        file_registry_id=file_registry["id"],
        temp_file_path=temp_file_path
    )
    
    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "documents_added": 0,
        "success": True,
        "file_path": temp_file_path,
        "file_url": "",
        "original_filename": f"urls_{len(request.urls)}",
        "plugin_name": request.plugin_name,
        "file_registry_id": file_registry["id"],
        "status": "processing"
    }

@app.post(
    "/collections/{collection_id}/ingest-file",
    response_model=AddDocumentsResponse,
    summary="Ingest a file directly into a collection",
    description="Upload, process, and add a file to a collection in one operation.",
    tags=["Ingestion"]
)
async def ingest_file_to_collection(
    collection_id: int,
    file: UploadFile = File(...),
    plugin_name: str = Form(...),
    plugin_params: str = Form("{}"),
    token: str = Depends(verify_token),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    params = json.loads(plugin_params)
    
    collection = CollectionsService.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection with ID {collection_id} not found")
    
    plugin = IngestionService.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plugin {plugin_name} not found")
    
    collection_name = collection.name if hasattr(collection, 'name') else collection['name']
    owner = collection.owner if hasattr(collection, 'owner') else collection['owner']
    
    file_info = IngestionService.save_file(
        file_content=file.file,
        filename=file.filename,
        owner=owner,
        collection_name=collection_name,
        content_type=file.content_type
    )
    
    file_registry = IngestionRepository.register_file(
        collection_id=collection_id,
        file_path=file_info["file_path"],
        file_url=file_info["file_url"],
        original_filename=file_info["original_filename"],
        plugin_name=plugin_name,
        plugin_params=params,
        owner=owner,
        document_count=0,
        content_type=file.content_type,
        status=FileStatus.PROCESSING
    )
    
    if not file_registry:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register file")
    
    background_tasks.add_task(
        IngestionService.process_uploaded_file,
        file_path=file_info["file_path"],
        plugin_name=plugin_name,
        plugin_params=params,
        collection_id=collection_id,
        file_registry_id=file_registry["id"]
    )
    
    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "documents_added": 0,
        "success": True,
        "file_path": file_info["file_path"],
        "file_url": file_info["file_url"],
        "original_filename": file_info["original_filename"],
        "plugin_name": plugin_name,
        "file_registry_id": file_registry["id"],
        "status": "processing"
    }

@app.post(
    "/collections",
    response_model=CollectionResponse,
    summary="Create collection",
    description="Create a new knowledge base collection.",
    tags=["Collections"],
    status_code=status.HTTP_201_CREATED
)
async def create_collection(
    collection: CollectionCreate, 
    token: str = Depends(verify_token)
):
    return CollectionsService.create_collection(collection)

@app.get(
    "/collections",
    response_model=CollectionList,
    summary="List collections",
    description="List all available knowledge base collections.",
    tags=["Collections"]
)
async def list_collections(
    token: str = Depends(verify_token),
    skip: int = Query(0, ge=0, description="Number of collections to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of collections to return"),
    owner: str = Query(None, description="Filter by owner"),
    visibility: str = Query(None, description="Filter by visibility ('private' or 'public')")
):
    return CollectionsService.list_collections(
        skip=skip,
        limit=limit,
        owner=owner,
        visibility=visibility
    )

@app.get(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    summary="Get collection",
    description="Get details of a specific knowledge base collection.",
    tags=["Collections"]
)
async def get_collection(
    collection_id: int,
    token: str = Depends(verify_token)
):
    return CollectionsService.get_collection(collection_id)

@app.patch(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    summary="Update collection",
    description="Update details of a specific knowledge base collection.",
    tags=["Collections"]
)
async def update_collection(
    collection_id: int,
    update_data: CollectionUpdate,
    token: str = Depends(verify_token)
):
    embeddings_model = update_data.embeddings_model
    
    return CollectionsService.update_collection(
        collection_id=collection_id,
        name=update_data.name,
        description=update_data.description,
        visibility=update_data.visibility,
        model=embeddings_model.model if embeddings_model else None,
        vendor=embeddings_model.vendor if embeddings_model else None,
        endpoint=embeddings_model.api_endpoint if embeddings_model else None,
        apikey=embeddings_model.apikey if embeddings_model else None
    )

@app.get(
    "/collections/{collection_id}/files",
    response_model=List[FileRegistryResponse],
    summary="List files in a collection",
    description="Get a list of all files in a collection.",
    tags=["Files"]
)
async def list_files(
    collection_id: int,
    token: str = Depends(verify_token),
    status: str = Query(None, description="Filter by status (completed, processing, failed, deleted)")
):
    return CollectionsService.list_files(collection_id, status)

@app.get(
    "/files/{file_id}/content",
    summary="Get file content",
    description="Get the content of a file from the collection.",
    tags=["Files"]
)
async def get_file_content(
    file_id: int,
    token: str = Depends(verify_token)
):
    content = IngestionService.get_file_content(file_id)
    if not content:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File content with ID {file_id} not found")
    return content