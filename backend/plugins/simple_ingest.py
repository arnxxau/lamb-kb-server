"""
Simple ingestion plugin for text files.

This plugin handles plain text files (txt, md) with chunking using LangChain's RecursiveCharacterTextSplitter.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import IngestPlugin, PluginRegistry


@PluginRegistry.register
class SimpleIngestPlugin(IngestPlugin):
    """Plugin for ingesting simple text files with LangChain's RecursiveCharacterTextSplitter."""
    
    name = "simple_ingest"
    description = "Ingest text files with LangChain's RecursiveCharacterTextSplitter"
    supported_file_types = {"txt", "md", "markdown", "text"}
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters accepted by this plugin.
        
        Returns:
            A dictionary mapping parameter names to their specifications
        """
        return {
            "chunk_size": {
                "type": "integer",
                "description": "Size of each chunk (uses LangChain default if not specified)",
                "required": False
            },
            "chunk_overlap": {
                "type": "integer",
                "description": "Number of units to overlap between chunks (uses LangChain default if not specified)",
                "required": False
            }
        }
    
    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Ingest a text file and split it into chunks using LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            file_path: Path to the file to ingest
            chunk_size: Size of each chunk (default: uses LangChain default)
            chunk_overlap: Number of units to overlap between chunks (default: uses LangChain default)
            file_url: URL to access the file (default: None)
            
        Returns:
            A list of dictionaries, each containing:
                - text: The chunk text
                - metadata: A dictionary of metadata for the chunk
        """
        # Extract parameters
        chunk_size = kwargs.get("chunk_size", None)
        chunk_overlap = kwargs.get("chunk_overlap", None)
        file_url = kwargs.get("file_url", "")
        
        # Create parameters dict for RecursiveCharacterTextSplitter initialization
        splitter_params = {}
        if chunk_size is not None:
            splitter_params["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            splitter_params["chunk_overlap"] = chunk_overlap
        
        # Read the file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise
        
        # Get file metadata
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        file_extension = file_path_obj.suffix.lstrip(".")
        file_size = os.path.getsize(file_path)
        
        # Create base metadata
        base_metadata = {
            "source": file_path,
            "filename": file_name,
            "extension": file_extension,
            "file_size": file_size,
            "file_url": file_url,
            "chunking_strategy": "langchain_recursive_character"
        }
        
        # Add chunking parameters to metadata if provided
        if chunk_size is not None:
            base_metadata["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            base_metadata["chunk_overlap"] = chunk_overlap
        
        # Create RecursiveCharacterTextSplitter with default or provided parameters
        text_splitter = RecursiveCharacterTextSplitter(**splitter_params)
        
        # Split content into chunks using LangChain
        try:
            chunks = text_splitter.split_text(content)
        except Exception as e:
            raise ValueError(f"Error splitting content into chunks: {str(e)}")
        
        # Create result documents with metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_count": len(chunks)
            })
            
            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return result