"""
URL ingestion plugin for web pages.

This plugin handles URLs by fetching their content and processing them into chunks.
Uses Firecrawl Python SDK for web scraping and crawling.
Supports both cloud and local Firecrawl instances.
"""

import os
import time
from typing import Dict, List, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Firecrawl configuration from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "")

# Import Firecrawl SDK
from firecrawl.firecrawl import FirecrawlApp

# Import chunking utilities
from .chunking import ChunkUnit, split_content
from .base import IngestPlugin, PluginRegistry


@PluginRegistry.register
class URLIngestPlugin(IngestPlugin):
    """Plugin for ingesting web pages from URLs using Firecrawl."""
    
    name = "url_ingest"
    description = "Ingest web pages from URLs using Firecrawl"
    supported_file_types = {"url"}
    
    def __init__(self):
        """Initialize the plugin with Firecrawl app."""
        super().__init__()
        # Initialize Firecrawl app
        self.firecrawl_app = self._init_firecrawl()
    
    def _init_firecrawl(self):
        """Initialize Firecrawl app which will use environment variables automatically."""
        try:
            # Log what configuration we're expecting based on environment variables
            if FIRECRAWL_API_URL:
                print(f"INFO: [url_ingest] Initializing Firecrawl with custom URL from environment: {FIRECRAWL_API_URL}")
                if FIRECRAWL_API_KEY:
                    print(f"INFO: [url_ingest] API key is also provided in environment variables")
            elif FIRECRAWL_API_KEY:
                print(f"INFO: [url_ingest] Initializing Firecrawl with API key from environment (cloud service)")
            else:
                print(f"INFO: [url_ingest] Initializing Firecrawl with default configuration")
                
            # Let FirecrawlApp handle environment variables internally
            return FirecrawlApp()
        except Exception as e:
            print(f"ERROR: [url_ingest] Failed to initialize Firecrawl: {str(e)}")
            raise ImportError(f"Firecrawl SDK required. Please install with: pip install firecrawl-py")
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters accepted by this plugin.
        
        Returns:
            A dictionary mapping parameter names to their specifications
        """
        return {
            "chunk_size": {
                "type": "integer",
                "description": "Size of each chunk",
                "default": 1000,
                "required": False
            },
            "chunk_unit": {
                "type": "string",
                "description": "Unit for chunking (char, word, line)",
                "enum": ["char", "word", "line"],
                "default": "char",
                "required": False
            },
            "chunk_overlap": {
                "type": "integer",
                "description": "Number of units to overlap between chunks",
                "default": 200,
                "required": False
            },
            "urls": {
                "type": "array",
                "description": "List of URLs to ingest",
                "required": True
            }
        }
    
    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Ingest URLs and split content into chunks using batch processing exclusively.
        
        Args:
            file_path: Path to a text file containing URLs or a placeholder (not used)
            urls: List of URLs to ingest
            chunk_size: Size of each chunk (default: 1000)
            chunk_unit: Unit for chunking - char, word, or line (default: char)
            chunk_overlap: Number of units to overlap between chunks (default: 200)
            
        Returns:
            A list of dictionaries, each containing:
                - text: The chunk text
                - metadata: A dictionary of metadata for the chunk
        """
        # Extract parameters with defaults
        chunk_size = kwargs.get("chunk_size", 1000)
        chunk_unit = ChunkUnit(kwargs.get("chunk_unit", "char"))
        chunk_overlap = kwargs.get("chunk_overlap", 200)
        urls = kwargs.get("urls", [])
        
        print(f"INFO: [url_ingest] Ingesting {len(urls)} URLs with chunk_size={chunk_size}, chunk_unit={chunk_unit}, chunk_overlap={chunk_overlap}")
        
        if not urls:
            raise ValueError("No URLs provided. Please provide a list of URLs to ingest.")
        
        # Ensure urls is a list
        if isinstance(urls, str):
            urls = [urls]
        
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        all_documents = []
        
        # Always use batch processing, even for a single URL
        print(f"INFO: [url_ingest] Starting batch scrape for {len(urls)} URLs")
        batch_params = {
            'formats': ['markdown'],
            'pageOptions': {
                'onlyMainContent': True
            }
        }
        
        start_time = time.time()
        batch_result = self.firecrawl_app.batch_scrape_urls(urls, batch_params)
        elapsed_time = time.time() - start_time
        
        print(f"INFO: [url_ingest] Batch scrape completed in {elapsed_time:.2f} seconds")
        print(f"INFO: [url_ingest] Batch response status: {batch_result.get('status')}")
        print(f"INFO: [url_ingest] Batch response data count: {len(batch_result.get('data', []))}")
        
        # Process batch results
        if batch_result and "data" in batch_result and batch_result.get("status") == "completed":
            for i, result in enumerate(batch_result["data"]):
                url = urls[i] if i < len(urls) else "unknown_url"
                try:
                    # Extract markdown content from the result
                    content = None
                    if "markdown" in result:
                        content = result["markdown"]
                        content_length = len(content)
                        print(f"INFO: [url_ingest] URL {url} content extracted: {content_length} chars")
                    
                    if content:
                        print(f"INFO: [url_ingest] Processing content for URL: {url}")
                        # Create base metadata for this URL
                        base_metadata = {
                            "source": url,
                            "filename": url,
                            "extension": "url",
                            "file_size": len(content),
                            "file_url": url,
                            "chunking_strategy": self.name,
                            "chunk_unit": str(chunk_unit),
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap
                        }
                        
                        # Split content into chunks using imported function
                        chunks = split_content(content, chunk_size, chunk_unit, chunk_overlap)
                        print(f"INFO: [url_ingest] Content split into {len(chunks)} chunks")
                        
                        # Create result documents with metadata
                        for j, chunk_text in enumerate(chunks):
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "chunk_index": j,
                                "chunk_count": len(chunks)
                            })
                            
                            all_documents.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                    else:
                        print(f"WARNING: [url_ingest] No markdown content found for URL {url} in batch response")
                except Exception as e:
                    print(f"ERROR: [url_ingest] Failed to process batch result for URL {url}: {str(e)}")
                    continue
        else:
            error_msg = f"Batch processing failed with status: {batch_result.get('status', 'unknown')}"
            print(f"ERROR: [url_ingest] {error_msg}")
            raise ValueError(error_msg)
        
        print(f"INFO: [url_ingest] Completed processing for {len(urls)} URLs, generated {len(all_documents)} document chunks")
        return all_documents