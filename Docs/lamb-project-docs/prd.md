# Lamb Project Requirements Document

## Project Description

The LAMB project is a web application designed to help educators manage their learning assistants (LAs) and to provide a platform for students to access and manage their LA's.

LAMB provides a User interface to creators (educators) wich will allow:
- Create, update, and delete LAs as needed
- Create, update, and delete Knowledge Bases (KBs) as needed
- Test and refine assistant responses
- Publish assistants as LTI activities

## LAMB structure 

### LAMB's Assitants 

LAMB's Assitants are provided via a Webservices API which allows for integration with various tools and platforms. This webservices API is compliant with Open AI API specifications to access LLMs.

This mainly can be found at the endpoints:
- GET /models
- GET /v1/models
- POST /chat/completions
- POST /v1/chat/completions


### LAMB integration with Open WebUi

LAMB is integrated with the Web based Chatbot interface Open Webui (OWI). Open Webui (https://github.com/open-webui) is a web based chatbot interface that allows users to interact with LLMs via a web based interface. LAMB is deeply integrated with Open Webui, specifically:
- User management and authentication
- Management of access permissions per assistant (named Models on Open WebUI)
- Knowledge Bases (KBs) management (LAMB uses the internal Open Webui KB to provide context and support to the assistant). 

This means that LAMB manages OWI´s users, groups, models, KBs, and other features directly. To do that, LAMB uses OWI's REST API to interact with OWI's components, and a couple of custom modifications to OWI's code.


### LAMB's INtegration with Learning Management Systems via IMS LTI

LAMB allows the educators to publish their Learning Assistants as IMS LTI 1.1 activities. This means the educator will be able to embed a specific Learning Assistant as a tool in their learning management system of choice. For instance, an educator can create an assitant with access to all the lecture notes of his course and give his students access to this assistant as a learning activity on his course on Moodle, using Moodle's External Tool component.


## Lamb Tech Stack 

LAMB is built using the following technologies:

### Backend 

- **Python 3.11** as the primary programming language
- **FastAPI** for building high-performance RESTful APIs
- **Uvicorn** as the ASGI server implementation
- **Pydantic** for data validation and settings management
- **SQLAlchemy** and **SQLModel** for ORM and database interactions
- **SQLite** for local database storage
- **JWT** with PassLib for authentication and security
- **Requests** and **AIOHTTP** for HTTP client operations

### AI and ML Libraries

- **OpenAI**, **Anthropic**, and **Google Generative AI** SDKs for LLM integration
- **Sentence-Transformers** and **HuggingFace Transformers** for NLP tasks
- **LlamaIndex** for RAG (Retrieval-Augmented Generation) capabilities
- **ChromaDB** for vector database functionality
- **PyTorch**, **NumPy**, and **Pandas** for ML operations

### Frontend

- **Svelte 5** as the frontend framework
- **SvelteKit** for server-side rendering and routing
- **TailwindCSS** for styling and UI components
- **Axios** for HTTP requests
- **svelte-i18n** for internationalization

### DevOps and Deployment

- **Docker** for containerization and deployment
- **Git** for version control

### Integrations

- **IMS LTI 1.1** for Learning Management System integration
- **Open WebUI** for chatbot interface integration
- **REST API** compatibility with OpenAI API specifications
