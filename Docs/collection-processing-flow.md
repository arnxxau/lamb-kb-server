# LAMB Knowledge Base Server: Collection, Ingestion, and Query Flow

This document provides a comprehensive overview of how collections are created, documents are ingested, and queries are processed in the LAMB Knowledge Base server.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Collection Lifecycle](#collection-lifecycle)
4. [Document Ingestion Flow](#document-ingestion-flow)
5. [Query Flow](#query-flow)
6. [Embedding Model Consistency](#embedding-model-consistency)
7. [Error Handling and Recovery](#error-handling-and-recovery)
8. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Architecture Overview

The LAMB Knowledge Base server provides vector database functionality using ChromaDB as the underlying vector store. The system consists of:

- **FastAPI Backend**: Handles API requests, routes them to appropriate services
- **SQLite Database**: Stores metadata about collections, files, and configuration
- **ChromaDB**: Stores document vectors and enables similarity search
- **Embedding Models**: Creates vector embeddings from text (via various providers)
- **Plugin System**: Enables extensible document ingestion and querying

The architecture follows a layered design:

1. **API Layer** (`main.py`): Defines endpoints and handles HTTP requests
2. **Service Layer** (`services/`): Contains business logic for collections, ingestion, and querying
3. **Data Access Layer** (`database/`): Manages connections and operations with SQLite and ChromaDB
4. **Plugin Layer** (`plugins/`): Provides extensible functionality for ingestion and querying

## Key Components

### Files and Directories

- `backend/main.py`: Main application entry point, API endpoints definition
- `backend/database/`: Database connections and models
  - `connection.py`: Database connection management and embedding function creation
  - `models.py`: SQLAlchemy models for collections and files
  - `service.py`: Data access methods for collections
- `backend/services/`: Business logic
  - `collections.py`: Collection management logic
  - `ingestion.py`: Document ingestion logic
  - `query.py`: Query processing logic
- `backend/plugins/`: Extensible plugins
  - `simple_ingest.py`: Default ingestion plugin
  - `simple_query.py`: Default query plugin
  - `base.py`: Plugin registration and interface definitions
- `backend/schemas/`: Pydantic models for request/response validation

### Key Classes and Components

1. **CollectionService** (`database/service.py`): Handles low-level collection operations
2. **CollectionsService** (`services/collections.py`): Provides API-level collection operations
3. **IngestionService** (`services/ingestion.py`): Manages document ingestion
4. **QueryService** (`services/query.py`): Processes queries against collections
5. **PluginRegistry** (`plugins/base.py`): Manages registration and discovery of plugins

## Collection Lifecycle

### Collection Creation Flow

1. **API Endpoint**: `POST /collections` in `main.py`
2. **Service**: `CollectionsService.create_collection` in `services/collections.py`
3. **Database Operation**: `CollectionService.create_collection` in `database/service.py`

The collection creation process:

1. Client sends a `CollectionCreate` request with:
   - Collection name, description, owner, visibility
   - Embeddings model configuration (vendor, model, API key, endpoint)

2. `main.py` resolves "default" values in embeddings model configuration by checking environment variables

3. `CollectionsService.create_collection`:
   - Validates the collection doesn't already exist
   - Converts visibility string to enum
   - Validates embeddings model if non-default values are provided
   - Tests the embedding function with a simple text to verify it works

4. `CollectionService.create_collection`:
   - Creates ChromaDB collection with embedding function
   - Retrieves and stores the ChromaDB UUID (added in v0.6.0+)
   - Creates SQLite record with collection details and the ChromaDB UUID
   - Stores bidirectional references between SQLite and ChromaDB:
     - SQLite record contains ChromaDB UUID
     - ChromaDB metadata contains SQLite ID (when possible)
   - Cleans up SQLite record if ChromaDB creation fails

5. `CollectionsService.create_collection`:
   - Verifies that the ChromaDB UUID was properly stored
   - Returns HTTP 500 if UUID wasn't stored (indicating critical failure)
   - Returns the created collection details to the client

### Collection Retrieval Flow

1. **API Endpoint**: `GET /collections/{collection_id}` in `main.py`
2. **Service**: `CollectionsService.get_collection` in `services/collections.py`
3. **Database Operation**: `CollectionService.get_collection` in `database/service.py`

The collection retrieval process:

1. Client requests a collection by its SQLite ID
2. System retrieves the collection record from SQLite
3. Collection information includes the ChromaDB UUID
4. For operations that need ChromaDB access, the system can use:
   - Primary approach: Get collection directly using UUID with `get_collection(id=uuid)`
   - Fallback approach: Get collection by name with `get_collection(name=name)`

## Document Ingestion Flow

The system supports multiple ingestion paths:

### Path 1: Separate Upload and Ingestion

1. **Upload File**: `POST /collections/{collection_id}/upload`
   - Saves file to disk in the collection's directory
   - Returns file path for later ingestion

2. **Process File**: `POST /collections/{collection_id}/ingest`
   - Uses specified ingestion plugin to process file
   - Returns document chunks but doesn't add to collection

3. **Add Documents**: `POST /collections/{collection_id}/documents`
   - Adds processed documents to ChromaDB collection
   - Updates document registry

### Path 2: Combined Upload and Ingestion

1. **Ingest File Directly**: `POST /collections/{collection_id}/ingest-file`
   - Uploads file, processes it, and adds documents in a single operation
   - Combines all three steps from Path 1
   - Creates file registry entry automatically

### Ingestion Process Detail

When ingesting documents (`IngestionService.add_documents_to_collection`):

1. Get collection from SQLite database
2. Extract embedding configuration from collection record
3. Create embedding function using `get_embedding_function`
4. Verify ChromaDB collection exists (fail if missing)
5. Get ChromaDB collection using UUID or name with embedding function
6. Prepare documents with appropriate metadata
7. Add documents to ChromaDB collection in batches
8. Return information about documents added

Key files involved:
- `services/ingestion.py`: `add_documents_to_collection` method
- `database/connection.py`: `get_embedding_function` method
- `plugins/simple_ingest.py`: Default ingestion plugin

## Query Flow

### Query Process

1. **API Endpoint**: `POST /collections/{collection_id}/query` in `main.py`
2. **Service**: `QueryService.query_collection` in `services/query.py`
3. **Plugin**: `SimpleQueryPlugin.query` in `plugins/simple_query.py`

When querying a collection:

1. QueryService gets collection from SQLite database
2. Creates embedding function using the same collection record
3. Verifies ChromaDB collection exists (fail if missing)
4. Gets ChromaDB collection using UUID or name with embedding function
5. Passes collection, embedding function, and parameters to query plugin
6. Query plugin performs ChromaDB query with consistent embedding function
7. Results are formatted and returned

Key files involved:
- `services/query.py`: `query_collection` method
- `plugins/simple_query.py`: Default query plugin
- `database/connection.py`: `get_embedding_function` method

## Embedding Model Consistency

The system ensures embedding model consistency across all operations:

1. **Collection Creation**: 
   - Default values are resolved in `main.py`
   - Embedding function is created and tested during collection creation

2. **Document Ingestion**:
   - Embedding function is created based on SQLite collection record
   - The same embedding function is used for all documents in a collection
   - ChromaDB collection is retrieved with the same embedding function

3. **Querying**:
   - Embedding function is created based on the same SQLite collection record
   - ChromaDB collection is retrieved with the same embedding function
   - Same embedding model is used for queries as for document ingestion

All embedding functions flow from a single source of truth: the embeddings_model field in the Collection SQLite record.

### Embedding Function Creation

The embedding function is created in `database/connection.py`:

1. `get_embedding_function` retrieves the collection record
2. Extracts embedding configuration (vendor, model, API key, endpoint)
3. Creates embedding function using `get_embedding_function_by_params`
4. Tests the embedding function before returning it

## Error Handling and Recovery

The system includes several error handling mechanisms:

1. **Collection Integrity**:
   - If ChromaDB collection creation fails, SQLite record is deleted
   - If SQLite collection exists but ChromaDB collection doesn't, ingestion fails with clear error
   - Test document is added during collection creation to verify embedding works
   - ChromaDB UUID is stored in SQLite and verified after collection creation

2. **Embedding Consistency**:
   - Embedding functions are tested before use
   - Warning is logged if there's a mismatch between SQLite and ChromaDB configurations
   - Default values are properly resolved and validated

3. **Ingestion Error Handling**:
   - Documents are added in batches to limit impact of failures
   - Specific error messages for API key, timeout, and other issues
   - File status tracking allows retrying failed operations

4. **Collection Synchronization**:
   - ChromaDB UUID is stored in SQLite as `chromadb_uuid` to maintain direct link
   - System can retrieve collections using either SQLite ID or ChromaDB UUID
   - Diagnostics can detect UUID mismatches between databases

## Debugging and Troubleshooting

The system includes extensive debugging support:

1. **Logging**: Detailed logs throughout all operations
2. **Embedding Information**: Added to document metadata and query results
3. **Validation**: Checks at each stage of the process
4. **Explicit Errors**: Clear error messages for common issues
5. **UUID Diagnostics**: Tools to detect UUID mismatches and orphaned collections

### Common Issues and Solutions

1. **Embedding Dimension Mismatch**: Occurs when embeddings are created with different models
   - Solution: Recreate collection and re-ingest documents with consistent model

2. **Missing ChromaDB Collection**: SQLite record exists but ChromaDB collection doesn't
   - Solution: Recreate collection (can't recreate just ChromaDB part)

3. **Embedding API Issues**: Problems with third-party embedding services
   - Solution: Check API keys, endpoints, and service availability

4. **Default Value Resolution**: Environment variables not set for default values
   - Solution: Set required environment variables or provide explicit values

5. **UUID Mismatch**: ChromaDB UUID doesn't match what's stored in SQLite
   - Solution: Use diagnostics tools to identify mismatches, then recreate affected collections

## Conclusion

The LAMB Knowledge Base server provides a robust system for creating collections, ingesting documents, and querying using vector embeddings. The architecture ensures embedding model consistency throughout the lifecycle of collections and documents, with appropriate error handling and validation at each step. 

The ChromaDB UUID tracking feature provides a more direct and reliable link between SQLite and ChromaDB, improving resilience to collection naming conflicts and enhancing system integrity. This is especially important when working with ChromaDB v0.6.0+, where collections are primarily identified by UUID rather than name. 