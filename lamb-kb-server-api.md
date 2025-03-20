# Lamb Knowledge Base Server Documentation
Version: 0.1.0

A dedicated knowledge base server designed to provide robust vector database functionality 
    for the LAMB project and to serve as a Model Context Protocol (MCP) server.
    
    ## Authentication
    
    All API endpoints are secured with Bearer token authentication. The token must match 
    the `LAMB_API_KEY` environment variable (default: `0p3n-w3bu!`).
    
    Example:
    ```
    curl -H 'Authorization: Bearer 0p3n-w3bu!' http://localhost:9090/
    ```
    
    ## Features
    
    - Knowledge base management for LAMB Learning Assistants
    - Vector database services using ChromaDB
    - API access for the LAMB project
    - Model Context Protocol (MCP) compatibility
    

## Base URL
http://localhost:9090

## Authentication
All authenticated endpoints require a Bearer token for authentication. The token must match the `LAMB_API_KEY` environment variable.

```bash
Authorization: Bearer 0p3n-w3bu!
```

## Query
### List available query plugins
**GET** `/query/plugins`

Get a list of available query plugins.
    
    This endpoint returns a list of all registered query plugins with their metadata.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/query/plugins'       -H 'Authorization: Bearer 0p3n-w3bu!'
    ```

#### Responses
**200**: List of available query plugins
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

---

### Query a collection
**POST** `/collections/{collection_id}/query`

Query a collection using a specified plugin.
    
    This endpoint performs a query on a collection using the specified query plugin.
    
    Example for simple_query plugin:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/query'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "query_text": "What is the capital of France?",
        "top_k": 5,
        "threshold": 0.5,
        "plugin_params": {}
      }'
    ```
    
    Parameters for simple_query plugin:
    - query_text: The text to query for
    - top_k: Number of results to return (default: 5)
    - threshold: Minimum similarity threshold (0-1) (default: 0.0)

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |
| plugin_name | query | string | No | Name of the query plugin to use |

#### Request Body
Content-Type: `application/json`

Schema: `QueryRequest`

Example:
```bash
    curl -X POST 'http://localhost:9090/collections/1/query'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "query_text": "What is the capital of France?",
        "top_k": 5,
        "threshold": 0.5,
        "plugin_params": {}
      }'
    ```

#### Responses
**200**: Query results
Content-Type: `application/json`

**400**: Invalid query parameters

**401**: Unauthorized - Invalid or missing authentication token

**404**: Collection or query plugin not found

**422**: Validation Error
Content-Type: `application/json`

---

## System
### Root endpoint
**GET** `/`

Returns a welcome message to confirm the server is running.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/'       -H 'Authorization: Bearer 0p3n-w3bu!'
    ```

#### Responses
**200**: Successful response with welcome message
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

---

### Health check
**GET** `/health`

Check the health status of the server.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/health'
    ```

#### Responses
**200**: Server is healthy and running
Content-Type: `application/json`

---

## Database
### Database status
**GET** `/database/status`

Check the status of all databases.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/database/status'       -H 'Authorization: Bearer 0p3n-w3bu!'
    ```

#### Responses
**200**: Database status information
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

---

## Ingestion
### List ingestion plugins
**GET** `/ingestion/plugins`

List all available document ingestion plugins.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/ingestion/plugins'       -H 'Authorization: Bearer 0p3n-w3bu!'
    ```

#### Responses
**200**: List of available ingestion plugins
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

---

### Upload a file to a collection
**POST** `/collections/{collection_id}/upload`

Upload a file to a collection for later ingestion.
    
    This endpoint uploads a file to the server but does not process it yet. 
    The file will be stored in the collection's directory.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/upload'       -H 'Authorization: Bearer 0p3n-w3bu!'       -F 'file=@/path/to/your/document.txt'
    ```

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |

#### Request Body
Content-Type: `multipart/form-data`

Schema: `Body_upload_file_collections__collection_id__upload_post`

Example:
```bash
    curl -X POST 'http://localhost:9090/collections/1/upload'       -H 'Authorization: Bearer 0p3n-w3bu!'       -F 'file=@/path/to/your/document.txt'
    ```

#### Responses
**200**: File uploaded successfully
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

**404**: Collection not found

**422**: Validation Error
Content-Type: `application/json`

---

### Ingest a file with specified plugin
**POST** `/collections/{collection_id}/ingest`

Process a previously uploaded file using a specified ingestion plugin.
    
    This endpoint processes an uploaded file using the specified ingestion plugin
    but does not add the processed documents to the collection yet.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "file_path": "/path/to/uploaded/file.txt",
        "plugin_name": "simple_ingest",
        "plugin_params": {
          "chunk_size": 1000,
          "chunk_unit": "char",
          "chunk_overlap": 200
        }
      }'
    ```

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |

#### Request Body
Content-Type: `application/x-www-form-urlencoded`

Schema: `Body_ingest_file_collections__collection_id__ingest_post`

Example:
```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "file_path": "/path/to/uploaded/file.txt",
        "plugin_name": "simple_ingest",
        "plugin_params": {
          "chunk_size": 1000,
          "chunk_unit": "char",
          "chunk_overlap": 200
        }
      }'
    ```

#### Responses
**200**: File processed successfully
Content-Type: `application/json`

**400**: Invalid plugin parameters

**401**: Unauthorized - Invalid or missing authentication token

**404**: Collection or plugin not found

**422**: Validation Error
Content-Type: `application/json`

---

### Add documents to a collection
**POST** `/collections/{collection_id}/documents`

Add processed documents to a collection.
    
    This endpoint adds processed documents to a ChromaDB collection.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/documents'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "documents": [
          {
            "text": "Document content here...",
            "metadata": {
              "source": "file.txt",
              "chunk_index": 0
            }
          }
        ]
      }'
    ```

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |

#### Request Body
Content-Type: `application/json`

Schema: `AddDocumentsRequest`

Example:
```bash
    curl -X POST 'http://localhost:9090/collections/1/documents'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "documents": [
          {
            "text": "Document content here...",
            "metadata": {
              "source": "file.txt",
              "chunk_index": 0
            }
          }
        ]
      }'
    ```

#### Responses
**200**: Documents added successfully
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

**404**: Collection not found

**422**: Validation Error
Content-Type: `application/json`

---

### Ingest a file directly into a collection
**POST** `/collections/{collection_id}/ingest-file`

Upload, process, and add a file to a collection in one operation.
    
    This endpoint combines file upload, processing with an ingestion plugin, and adding 
    to the collection in a single operation.
    
    Example for simple_ingest plugin:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest-file'       -H 'Authorization: Bearer 0p3n-w3bu!'       -F 'file=@/path/to/document.txt'       -F 'plugin_name=simple_ingest'       -F 'plugin_params={"chunk_size": 1000, "chunk_unit": "char", "chunk_overlap": 200}'
    ```
    
    Parameters for simple_ingest plugin:
    - chunk_size: Size of each chunk (default: 1000)
    - chunk_unit: Unit for chunking (char, word, line) (default: char)
    - chunk_overlap: Number of units to overlap between chunks (default: 200)

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |

#### Request Body
Content-Type: `multipart/form-data`

Schema: `Body_ingest_file_to_collection_collections__collection_id__ingest_file_post`

Example:
```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest-file'       -H 'Authorization: Bearer 0p3n-w3bu!'       -F 'file=@/path/to/document.txt'       -F 'plugin_name=simple_ingest'       -F 'plugin_params={"chunk_size": 1000, "chunk_unit": "char", "chunk_overlap": 200}'
    ```

#### Responses
**200**: File ingested successfully
Content-Type: `application/json`

**400**: Invalid plugin parameters

**401**: Unauthorized - Invalid or missing authentication token

**404**: Collection or plugin not found

**500**: Error processing file or adding to collection

**422**: Validation Error
Content-Type: `application/json`

---

## Collections
### Create collection
**POST** `/collections`

Create a new knowledge base collection.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "name": "my-knowledge-base",
        "description": "My first knowledge base",
        "owner": "user1",
        "visibility": "private",
        "embeddings_model": {
          "model": "default",
          "vendor": "default",
          "apikey": "default"
        }
        
        # For OpenAI embeddings, use:
        # "embeddings_model": {
        #   "model": "text-embedding-3-small",
        #   "vendor": "openai",
        #   "apikey": "your-openai-key-here"
        # }
      }'
    ```

#### Request Body
Content-Type: `application/json`

Schema: `CollectionCreate`

Example:
```bash
    curl -X POST 'http://localhost:9090/collections'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "name": "my-knowledge-base",
        "description": "My first knowledge base",
        "owner": "user1",
        "visibility": "private",
        "embeddings_model": {
          "model": "default",
          "vendor": "default",
          "apikey": "default"
        }
        
        # For OpenAI embeddings, use:
        # "embeddings_model": {
        #   "model": "text-embedding-3-small",
        #   "vendor": "openai",
        #   "apikey": "your-openai-key-here"
        # }
      }'
    ```

#### Responses
**201**: Collection created successfully
Content-Type: `application/json`

**400**: Bad request - Invalid collection data

**409**: Conflict - Collection with this name already exists

**401**: Unauthorized - Invalid or missing authentication token

**422**: Validation Error
Content-Type: `application/json`

---

### List collections
**GET** `/collections`

List all available knowledge base collections.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/collections'       -H 'Authorization: Bearer 0p3n-w3bu!'
    
    # With filtering parameters
    curl -X GET 'http://localhost:9090/collections?owner=user1&visibility=public&skip=0&limit=20'       -H 'Authorization: Bearer 0p3n-w3bu!'
    ```

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| skip | query | integer | No | Number of collections to skip |
| limit | query | integer | No | Maximum number of collections to return |
| owner | query | string | No | Filter by owner |
| visibility | query | string | No | Filter by visibility ('private' or 'public') |

#### Responses
**200**: List of collections
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

**422**: Validation Error
Content-Type: `application/json`

---

### Get collection
**GET** `/collections/{collection_id}`

Get details of a specific knowledge base collection.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/collections/1'       -H 'Authorization: Bearer 0p3n-w3bu!'
    ```

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |

#### Responses
**200**: Collection details
Content-Type: `application/json`

**404**: Not found - Collection not found

**401**: Unauthorized - Invalid or missing authentication token

**422**: Validation Error
Content-Type: `application/json`

---

## Files
### List files in a collection
**GET** `/collections/{collection_id}/files`

Get a list of all files in a collection

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| collection_id | path | integer | Yes |  |
| status | query | string | No | Filter by status (completed, processing, failed, deleted) |

#### Responses
**200**: List of files in the collection
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

**404**: Collection not found

**500**: Server error

**422**: Validation Error
Content-Type: `application/json`

---

### Update file status
**PUT** `/files/{file_id}/status`

Update the status of a file in the registry

#### Parameters
| Name | In | Type | Required | Description |
| ---- | -- | ---- | -------- | ----------- |
| file_id | path | integer | Yes |  |
| status | query | string | Yes | New status (completed, processing, failed, deleted) |

#### Responses
**200**: File status updated successfully
Content-Type: `application/json`

**401**: Unauthorized - Invalid or missing authentication token

**404**: File not found

**500**: Server error

**422**: Validation Error
Content-Type: `application/json`

---

## Models
### AddDocumentsRequest
Request to add documents to a collection.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| documents | array | List of documents to add |

---

### AddDocumentsResponse
Response with the results of adding documents to a collection.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| collection_id | integer | ID of the collection |
| collection_name | string | Name of the collection |
| documents_added | integer | Number of documents added |
| success | boolean | Whether the operation was successful |

---

### Body_ingest_file_collections__collection_id__ingest_post

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| file_path | string |  |
| plugin_name | string |  |
| plugin_params | string |  |

---

### Body_ingest_file_to_collection_collections__collection_id__ingest_file_post

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| file | string |  |
| plugin_name | string |  |
| plugin_params | string |  |

---

### Body_upload_file_collections__collection_id__upload_post

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| file | string |  |

---

### CollectionCreate
Schema for creating a new collection.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| name | string | Name of the collection |
| description | object | Optional description of the collection |
| visibility | string | Visibility setting ('private' or 'public') |
| owner | string | Owner of the collection |
| embeddings_model | object | Optional custom embeddings model configuration |

---

### CollectionList
Schema for list of collections response.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| total | integer | Total number of collections matching filters |
| items | array | List of collections |

---

### CollectionResponse
Schema for collection response.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| name | string | Name of the collection |
| description | object | Optional description of the collection |
| visibility | string | Visibility setting ('private' or 'public') |
| id | integer | Unique identifier of the collection |
| owner | string | Owner of the collection |
| creation_date | string | Creation date of the collection |
| embeddings_model | [EmbeddingsModel](#embeddingsmodel) | Embeddings model configuration |

---

### DatabaseStatusResponse
Model for database status response

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| sqlite_status | object | Status of SQLite database |
| chromadb_status | object | Status of ChromaDB database |
| collections_count | integer | Number of collections |

---

### Document
A document chunk with metadata.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| text | string | Text content of the document |
| metadata | [DocumentMetadata](#documentmetadata) | Metadata for the document |

---

### DocumentMetadata
Metadata for a document.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| source | string | Source of the document |
| filename | string | Name of the source file |
| extension | string | Extension of the source file |
| file_size | integer | Size of the source file in bytes |
| chunking_strategy | string | Strategy used for chunking |
| chunk_unit | string | Unit used for chunking |
| chunk_size | integer | Size of each chunk |
| chunk_overlap | integer | Overlap between chunks |
| chunk_index | integer | Index of this chunk |
| chunk_count | integer | Total number of chunks |
| document_id | object | ID of the document in ChromaDB |
| ingestion_timestamp | object | Timestamp when the document was ingested |

---

### EmbeddingsModel
Schema for embeddings model configuration.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| model | string | Name or path of the embeddings model |
| endpoint | object | Optional custom API endpoint |
| apikey | object | Optional API key for the endpoint |

---

### FileRegistryResponse
Model for file registry entry response

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| id | integer | ID of the file registry entry |
| collection_id | integer | ID of the collection |
| original_filename | string | Original filename |
| file_path | string | Path to the file on the server |
| file_url | string | URL to access the file |
| file_size | integer | Size of the file in bytes |
| content_type | object | MIME type of the file |
| plugin_name | string | Name of the ingestion plugin used |
| plugin_params | object | Parameters used for ingestion |
| status | string | Status of the file |
| document_count | integer | Number of documents created from this file |
| created_at | string | Timestamp when the file was added |
| updated_at | string | Timestamp when the file record was last updated |
| owner | string | Owner of the file |

---

### HTTPValidationError

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| detail | array |  |

---

### HealthResponse
Model for health check responses

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| status | string | Status of the server |
| version | string | Server version |

---

### IngestFileResponse
Response with the results of file ingestion.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| file_path | string | Path to the ingested file |
| document_count | integer | Number of document chunks created |
| documents | array | List of document chunks |

---

### IngestionPluginInfo
Information about an ingestion plugin.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| name | string | Name of the plugin |
| description | string | Description of the plugin |
| supported_file_types | array | File types supported by the plugin |
| parameters | object | Parameters accepted by the plugin |

---

### MessageResponse
Model for basic message responses

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| message | string | Response message |

---

### QueryPluginInfo
Information about a query plugin.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| name | string | Plugin name |
| description | string | Plugin description |
| parameters | object | Plugin parameters |

---

### QueryRequest
Query request schema.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| query_text | string | Text to query for |
| top_k | integer | Number of results to return |
| threshold | number | Minimum similarity threshold (0-1) |
| plugin_params | object | Additional plugin-specific parameters |

---

### QueryResponse
Query response schema.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| results | array | List of query results |
| count | integer | Number of results returned |
| timing | object | Timing information for the query |
| query | string | Original query text |

---

### QueryResult
Single query result item.

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| similarity | number | Similarity score (0-1) |
| data | string | Document content |
| metadata | object | Document metadata |

---

### ValidationError

#### Properties
| Name | Type | Description |
| ---- | ---- | ----------- |
| loc | array |  |
| msg | string |  |
| type | string |  |

---
