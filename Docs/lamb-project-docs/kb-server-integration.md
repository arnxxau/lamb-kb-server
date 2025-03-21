# Knowledge Base Server Integration

## Overview

This document provides a high-level conceptual overview of how LAMB integrates with the Knowledge Base (KB) server. For detailed API specifications, refer to the [KB Server API Documentation](kb-server-docs/lamb-kb-server-api.md).

## Basic Concepts

### Terminology

- **Knowledge Base/Collection**: These terms are used interchangeably in the current implementation. They represent a collection of documents that can be queried for information.
- **Document/File**: The content units that are stored within a knowledge base, which can be indexed and searched.

### Integration Architecture

The LAMB application communicates with a separate KB server that handles document storage, indexing, and retrieval. This separation of concerns allows:

1. **Scalability**: The KB server can be scaled independently of the core application
2. **Modularity**: The KB functionality can be maintained separately
3. **Flexibility**: Different KB implementations can be swapped if needed

```
┌────────────┐     HTTP/REST      ┌──────────────┐
│            │  ----------------> │              │
│  LAMB      │                    │  KB Server   │
│  Backend   │ <---------------- │              │
│            │   JSON Responses   │              │
└────────────┘                    └──────────────┘
       │                                  │
       │ API                              │ Storage/Indexing
       ▼                                  ▼
┌────────────┐                    ┌──────────────┐
│  Frontend  │                    │  Document    │
│  UI        │                    │  Storage     │
└────────────┘                    └──────────────┘
```

## Configuration

### Environment Variables

The integration relies on the following environment variables:

```
LAMB_KB_SERVER=http://localhost:9090        # KB server URL
LAMB_KB_SERVER_TOKEN=your-auth-token        # Authentication token
```

### Authentication

All requests to the KB server include an authentication token sent as a Bearer token in the Authorization header. This token must match the one configured in the KB server.

## Error Handling

### Server Availability

The LAMB backend checks the KB server's availability before attempting any operations and provides appropriate feedback:

1. If the KB server is unavailable, endpoints return a standardized error response:
   ```json
   {
     "status": "error",
     "message": "Knowledge Base server offline",
     "kb_server_available": false
   }
   ```

2. The frontend handles these errors by displaying user-friendly messages indicating that the KB server is offline.

### Best Practices

- Always check KB server availability before operations
- Provide clear user feedback when the server is unavailable
- Implement retries with exponential backoff for transient errors
- Log server connectivity issues for monitoring

## Usage Flow

1. **Authentication**: User authenticates with LAMB
2. **Operations**: LAMB backend forwards KB operations to the KB server
3. **Results**: KB server responds with operation results
4. **Presentation**: LAMB presents the results to the user

## Limitations

- Knowledge bases cannot currently be edited once created (only files can be added/removed)
- The KB server must be configured separately from the main application
- Offline operation is not supported - the KB features will be unavailable if the KB server is down

## Further Reading

For detailed API specifications and endpoint documentation, refer to the [KB Server API Documentation](kb-server-docs/lamb-kb-server-api.md).
