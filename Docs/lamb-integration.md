# LAMB Authentication Integration Guide

## Overview

This document provides technical details for integrating a sibling project with LAMB's authentication system. It allows separate applications to use LAMB's authentication mechanism, ensuring a unified user management approach across multiple applications.

## Authentication Architecture

LAMB employs a dual-layer authentication system:

1. **LAMB Creator User**: Primary user entities stored in the LAMB database
2. **OWI User**: Corresponding user entities in the Open WebUI system

```
┌─────────────────┐          ┌─────────────────┐
│                 │          │                 │
│  LAMB Creator   │  1:1     │    OWI User     │
│     User        │◄────────►│                 │
│                 │          │                 │
└─────────────────┘          └─────────────────┘
```

## Integration Approach

The integration follows these principles:

1. **Token-based Authentication**: JWT tokens issued by LAMB are used for cross-application authentication
2. **API Gateway Pattern**: The sibling application communicates with LAMB's Creator Interface API
3. **Stateless Authentication**: No shared session state, only token validation

## Prerequisites

Before integrating, ensure:

1. LAMB is fully operational with Creator Interface API accessible
2. Network connectivity between the sibling application and LAMB
3. Required environment variables are configured

## Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LAMB_API_HOST` | URL to the LAMB Creator Interface API | `http://localhost:8000` |
| `LAMB_API_BEARER_TOKEN` | Bearer token for server-to-server API authentication | `your-secure-token` |

## Authentication Endpoints

### 1. User Verification (Login)

Verifies user credentials and returns a JWT token.

**Endpoint:** `POST /creator_user/verify`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure-password"
}
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "name": "User Name",
    "email": "user@example.com",
    "launch_url": "http://owi-instance.com/launch?token=...",
    "user_id": "12345",
    "role": "user"
  },
  "error": null
}
```

**Response (Failure):**
```json
{
  "success": false,
  "error": "Invalid credentials",
  "data": null
}
```

### 2. Token Validation

Validates a JWT token and returns the associated user information.

**Endpoint:** `GET /creator_user/validate_token`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "name": "User Name",
    "email": "user@example.com",
    "user_id": "12345",
    "role": "user",
    "exp": 1679234567  // Token expiration timestamp
  },
  "error": null
}
```

**Response (Failure):**
```json
{
  "success": false,
  "error": "Token expired or invalid",
  "data": null
}
```

## Integration Process

### 1. Authentication Flow

The integration process follows this sequence:

1. User attempts to log in to the sibling application
2. Sibling app forwards credentials to LAMB Creator Interface API
3. LAMB verifies credentials and returns a JWT token
4. Sibling app stores the token and uses it for subsequent requests
5. For protected resources, sibling app validates the token with LAMB
6. If token is valid, access is granted; otherwise, user is redirected to login

```mermaid
sequenceDiagram
    participant User
    participant SiblingApp
    participant LAMB
    participant OWI

    User->>SiblingApp: Login (email/password)
    SiblingApp->>LAMB: POST /creator_user/verify
    LAMB->>OWI: Verify credentials
    OWI-->>LAMB: Credentials valid
    LAMB-->>SiblingApp: JWT token
    SiblingApp-->>User: Login successful
    
    User->>SiblingApp: Access protected resource
    SiblingApp->>LAMB: GET /creator_user/validate_token
    LAMB-->>SiblingApp: Token validation result
    SiblingApp-->>User: Resource access granted/denied
```

### 2. Token Structure

The JWT token contains these key fields:

| Field | Description |
|-------|-------------|
| `sub` | Subject (user ID) |
| `name` | User's display name |
| `email` | User's email address |
| `role` | User's role (user/admin) |
| `exp` | Expiration timestamp |
| `iat` | Issued at timestamp |

## Implementation Examples

### Python Implementation (FastAPI)

```python
import httpx
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

class UserCredentials(BaseModel):
    email: str
    password: str

class AuthClient:
    def __init__(self):
        self.lamb_host = os.getenv("LAMB_API_HOST")
        self.lamb_bearer_token = os.getenv("LAMB_API_BEARER_TOKEN")
        
        if not self.lamb_host or not self.lamb_bearer_token:
            raise ValueError("LAMB_API_HOST and LAMB_API_BEARER_TOKEN environment variables are required")
    
    async def verify_user(self, email: str, password: str) -> Dict[str, Any]:
        """Verify user credentials against LAMB API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.lamb_host}/creator_user/verify",
                headers={
                    "Authorization": f"Bearer {self.lamb_bearer_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "email": email,
                    "password": password
                },
                timeout=30.0
            )
            
            return response.json()
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token against LAMB API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.lamb_host}/creator_user/validate_token",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            return response.json()

# Initialize auth client
auth_client = AuthClient()

# Login endpoint
@app.post("/login")
async def login(credentials: UserCredentials):
    result = await auth_client.verify_user(credentials.email, credentials.password)
    
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error", "Authentication failed"))
    
    return result["data"]

# Middleware for token validation
async def validate_auth_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    result = await auth_client.validate_token(token)
    
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid token"))
    
    return result["data"]

# Protected endpoint example
@app.get("/protected-resource")
async def protected_resource(user_data: dict = Depends(validate_auth_token)):
    return {
        "message": f"Hello, {user_data['name']}!",
        "data": "This is protected content",
        "user_info": user_data
    }
```

### JavaScript Implementation (Node.js with Express)

```javascript
const express = require('express');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(express.json());

// Auth client for LAMB API interaction
class AuthClient {
  constructor() {
    this.lambHost = process.env.LAMB_API_HOST;
    this.lambBearerToken = process.env.LAMB_API_BEARER_TOKEN;
    
    if (!this.lambHost || !this.lambBearerToken) {
      throw new Error('LAMB_API_HOST and LAMB_API_BEARER_TOKEN environment variables are required');
    }
  }
  
  async verifyUser(email, password) {
    try {
      const response = await axios.post(
        `${this.lambHost}/creator_user/verify`,
        { email, password },
        {
          headers: {
            'Authorization': `Bearer ${this.lambBearerToken}`,
            'Content-Type': 'application/json'
          },
          timeout: 30000
        }
      );
      
      return response.data;
    } catch (error) {
      console.error('Error verifying user:', error.message);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Server error during authentication'
      };
    }
  }
  
  async validateToken(token) {
    try {
      const response = await axios.get(
        `${this.lambHost}/creator_user/validate_token`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          timeout: 30000
        }
      );
      
      return response.data;
    } catch (error) {
      console.error('Error validating token:', error.message);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Server error during token validation'
      };
    }
  }
}

// Initialize auth client
const authClient = new AuthClient();

// Login endpoint
app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  
  if (!email || !password) {
    return res.status(400).json({ success: false, error: 'Email and password required' });
  }
  
  const result = await authClient.verifyUser(email, password);
  
  if (!result.success) {
    return res.status(401).json(result);
  }
  
  return res.json(result);
});

// Middleware for token validation
const validateAuthToken = async (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ success: false, error: 'Missing or invalid authorization header' });
  }
  
  const token = authHeader.replace('Bearer ', '');
  const result = await authClient.validateToken(token);
  
  if (!result.success) {
    return res.status(401).json(result);
  }
  
  req.user = result.data;
  next();
};

// Protected endpoint example
app.get('/protected-resource', validateAuthToken, (req, res) => {
  res.json({
    message: `Hello, ${req.user.name}!`,
    data: 'This is protected content',
    user_info: req.user
  });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
```

## Frontend Implementation (Svelte 5 Example)

```javascript
// auth.js - Authentication utilities
import axios from 'axios';

// Base API URL for your sibling application
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

// Storage keys
const TOKEN_KEY = 'auth_token';
const USER_KEY = 'user_data';

export const login = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/login`, { email, password });
    
    if (response.data.success) {
      // Store token and user data
      localStorage.setItem(TOKEN_KEY, response.data.data.token);
      localStorage.setItem(USER_KEY, JSON.stringify(response.data.data));
      return { success: true, data: response.data.data };
    } else {
      return { success: false, error: response.data.error };
    }
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data?.error || 'Failed to log in'
    };
  }
};

export const logout = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

export const getAuthToken = () => {
  return localStorage.getItem(TOKEN_KEY);
};

export const getUser = () => {
  const userData = localStorage.getItem(USER_KEY);
  return userData ? JSON.parse(userData) : null;
};

export const isAuthenticated = () => {
  return !!getAuthToken();
};

// Axios interceptor for adding auth token to requests
export const setupAxiosInterceptors = () => {
  axios.interceptors.request.use(
    config => {
      const token = getAuthToken();
      if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
      }
      return config;
    },
    error => {
      return Promise.reject(error);
    }
  );
  
  axios.interceptors.response.use(
    response => response,
    error => {
      if (error.response && error.response.status === 401) {
        // Token expired or invalid
        logout();
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }
  );
};
```

```svelte
<!-- Login.svelte - Login component -->
<script>
  import { login } from '../utils/auth';
  import { goto } from '$app/navigation';
  
  // Using Svelte 5's $state for reactive declarations
  let email = $state('');
  let password = $state('');
  let error = $state('');
  let loading = $state(false);
  
  async function handleSubmit(e) {
    e.preventDefault();
    loading = true;
    error = '';
    
    const result = await login(email, password);
    
    if (result.success) {
      goto('/dashboard');
    } else {
      error = result.error || 'Failed to log in';
    }
    
    loading = false;
  }
</script>

<div class="login-container">
  <h2>Login</h2>
  {#if error}
    <div class="error-message">{error}</div>
  {/if}
  <form on:submit={handleSubmit}>
    <div class="form-group">
      <label for="email">Email:</label>
      <input
        type="email"
        id="email"
        bind:value={email}
        required
      />
    </div>
    <div class="form-group">
      <label for="password">Password:</label>
      <input
        type="password"
        id="password"
        bind:value={password}
        required
      />
    </div>
    <button type="submit" disabled={loading}>
      {loading ? 'Logging in...' : 'Login'}
    </button>
  </form>
</div>

<style>
  .login-container {
    max-width: 400px;
    margin: 0 auto;
    padding: 2rem;
  }
  .error-message {
    color: red;
    margin-bottom: 1rem;
  }
  .form-group {
    margin-bottom: 1rem;
  }
  label {
    display: block;
    margin-bottom: 0.5rem;
  }
  input {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  button {
    background: #4CAF50;
    color: white;
    border: none;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    cursor: pointer;
  }
  button:disabled {
    background: #cccccc;
  }
</style>
```

## Security Considerations

1. **HTTPS Required**: All communication between applications must use HTTPS
2. **Token Storage**: Store tokens securely (use HttpOnly cookies when possible)
3. **Token Expiration**: Implement proper handling for token expiration
4. **Bearer Token Security**: Keep the LAMB_API_BEARER_TOKEN secure as it provides system-level access
5. **Rate Limiting**: Implement rate limiting to prevent brute force attacks
6. **Error Handling**: Never expose detailed system errors to clients

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check that credentials are correct
   - Ensure token hasn't expired
   - Verify LAMB_API_BEARER_TOKEN is correct

2. **Connection Refused**
   - Verify network connectivity between applications
   - Check that LAMB_API_HOST is correctly configured
   - Ensure LAMB Creator Interface API is running

3. **User Not Found**
   - Confirm the user exists in both LAMB and OWI systems
   - Check for typos in email addresses

### Debugging Tools

1. **Token Decoder**: Use [jwt.io](https://jwt.io/) to decode JWT tokens and inspect their contents
2. **API Testing**: Use tools like Postman or Insomnia to test API endpoints directly
3. **Logging**: Implement detailed logging for auth-related operations

## References

1. [JWT Introduction](https://jwt.io/introduction)
2. [OAuth 2.0 Authorization Framework](https://oauth.net/2/)
3. [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
