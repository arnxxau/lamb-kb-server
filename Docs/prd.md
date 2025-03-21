# Lamb Knowledge Base Server (lamb-kb-server) Requirements Document

## Project Description

The lamb-kb-server is a dedicated knowledge base server designed to provide robust vector database functionality for the LAMB project and to serve as a Model Context Protocol (MCP) server. It acts as an infrastructure component that enables retrieval-augmented generation capabilities for Learning Assistants (LAs) within the LAMB ecosystem.

## Purpose and Goals

The primary purposes of lamb-kb-server are:

1. **Knowledge Management**: Provide a repository for knowledge bases to be used by Learning Assistants in the LAMB project, or other projects using the API
2. **Vector Database Services**: Use ChromaDB vector storage and similarity search capabilities
3. **Database** Use SQLite for persistent storage of knowledge base data not related to embeddings
3. **API Access**: Expose a FastAPI-based  API that allows the LAMB project to interact with knowledge bases
4. **MCP Server**: Function as a Model Context Protocol server, allowing standardized access to contextual information by various language models
5. **Content Storage and Access**: Store documents and collections in a file system with proper access controls to be linked with URIs

## System Architecture

### Core Components

- **ChromaDB Database**: Vector database for storing and retrieving embeddings
- **SQLite Database**: Persistent storage for knowledge base metadata
- **FastAPI Server**: WS API framework for exposing knowledge base functionality
- **MCP Protocol Handler**: Component responsible for implementing the Model Context Protocol
- **Storage Manager**: Component for managing persistent storage of knowledge base data

### Integration Points

- **LAMB Project Integration**: Primary integration with the sibling LAMB project via Webservice 
- **Authentication**: Use JWT with PassLib for authentication and security, user management and authentication will be done on the LAMB /creator endpoints
- **Multiple Embeddings providers**: Embeddings will be provided by multiple providers. Setup by .env variables or the administrator

## Functional Requirements

### Knowledge Base Management

- Store and organize multiple knowledge bases
- Support for creation, updates, and deletion of knowledge bases
- Plugins for diferent strategies of ingestion (including but not limited to chuncking, metadata management, embeddings, and pre and post query processing  )
- Duplication of kbs and re-generate embeddings for documents


### API Functionality

### Document Repository

### MCP Server Functionality



## Technical Stack

### Backend

- **Python 3.11+** as the primary programming language
- **FastAPI** for building the REST API
- **Uvicorn** as the ASGI server implementation
- **Pydantic** for data validation and settings management
- **ChromaDB** for vector database functionality
- **SQLite** for persistent storage of knowledge base metadata

### Frontend

- **Svelte 5** as the primary frontend framework
- **JavaScript** for client-side logic
- **JDocs** for documentation
- **Tailwind CSS** for styling and UI components


### DevOps and Deployment

- **Docker** for containerization and deployment
- **Git** for version control

## Non-Functional Requirements

### Security

- Secure API access through authentication tokens
- Access control for different knowledge bases


### Reliability

- Comprehensive error handling and logging

### Maintainability

- Well-documented codebase
- Comprehensive API documentation

