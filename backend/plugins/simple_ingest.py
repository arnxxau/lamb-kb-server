"""
Simple ingestion plugin for text files.

This plugin handles plain text files (txt, md) with configurable chunking options.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import IngestPlugin, ChunkUnit, PluginRegistry


@PluginRegistry.register
class SimpleIngestPlugin(IngestPlugin):
    """Plugin for ingesting simple text files with configurable chunking."""
    
    name = "simple_ingest"
    description = "Ingest text files with configurable chunking options"
    supported_file_types = {"txt", "md", "markdown", "text"}
    
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
            }
        }
    
    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Ingest a text file and split it into chunks.
        
        Args:
            file_path: Path to the file to ingest
            chunk_size: Size of each chunk (default: 1000)
            chunk_unit: Unit for chunking - char, word, or line (default: char)
            chunk_overlap: Number of units to overlap between chunks (default: 200)
            file_url: URL to access the file (default: None)
            
        Returns:
            A list of dictionaries, each containing:
                - text: The chunk text
                - metadata: A dictionary of metadata for the chunk
        """
        # Extract parameters with defaults
        chunk_size = kwargs.get("chunk_size", 1000)
        chunk_unit = ChunkUnit(kwargs.get("chunk_unit", "char"))
        chunk_overlap = kwargs.get("chunk_overlap", 200)
        file_url = kwargs.get("file_url", "")
        
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
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
            "file_url": file_url,  # Include the URL to access the file
            "chunking_strategy": self.name,
            "chunk_unit": str(chunk_unit),  # Convert to string to ensure it's serializable
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        # Split content into chunks based on the specified unit and size
        try:
            chunks = self._split_content(content, chunk_size, chunk_unit, chunk_overlap)
        except Exception as e:
            raise
        
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
    
    def _split_content(self, content: str, chunk_size: int, 
                      chunk_unit: ChunkUnit, chunk_overlap: int) -> List[str]:
        """Split content into chunks based on the specified unit and size.
        
        Args:
            content: Text content to split
            chunk_size: Size of each chunk
            chunk_unit: Unit for chunking (char, word, line)
            chunk_overlap: Number of units to overlap between chunks
            
        Returns:
            List of content chunks
        """

        
        try:
            result = None
            if chunk_unit == ChunkUnit.CHAR:
                result = self._split_by_chars(content, chunk_size, chunk_overlap)
            elif chunk_unit == ChunkUnit.WORD:
                result = self._split_by_words(content, chunk_size, chunk_overlap)
            elif chunk_unit == ChunkUnit.LINE:
                result = self._split_by_lines(content, chunk_size, chunk_overlap)
            else:
                raise ValueError(f"Unsupported chunk unit: {chunk_unit}")
                
            return result
            
        except Exception as e:
            raise
    
    def _split_by_chars(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split content by characters.
        
        Args:
            content: Text content to split
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of characters to overlap
            
        Returns:
            List of content chunks
        """
        chunks = []
        start = 0
        content_len = len(content)
        
        iteration = 0
        
        while start < content_len:
            iteration += 1
            if iteration > 100:  # Safeguard against infinite loops
                break
                
            end = min(start + chunk_size, content_len)
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
        
        return chunks
    
    def _split_by_words(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split content by words.
        
        Args:
            content: Text content to split
            chunk_size: Number of words per chunk
            chunk_overlap: Number of words to overlap
            
        Returns:
            List of content chunks
        """
        # Split into words (considering punctuation and whitespace)
        words = re.findall(r'\S+|\s+', content)
        chunks = []
        
        start = 0
        words_len = len(words)
        
        iteration = 0
        
        while start < words_len:
            iteration += 1
            if iteration > 1000:  # Safeguard against infinite loops with a higher limit for words
                break
                
            end = min(start + chunk_size, words_len)
            chunk = ''.join(words[start:end])
            chunks.append(chunk)
            # Store the previous start position to detect lack of progress
            prev_start = start
            start = end - chunk_overlap
            
            # Handle negative or unchanged start value to prevent infinite loops
            if start < 0 or (start >= end and start < words_len) or start == prev_start:
                break
        
        return chunks
    
    def _split_by_lines(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split content by lines.
        
        Args:
            content: Text content to split
            chunk_size: Number of lines per chunk
            chunk_overlap: Number of lines to overlap
            
        Returns:
            List of content chunks
        """
        lines = content.splitlines(keepends=True)  # Keep line endings
        chunks = []
        
        start = 0
        lines_len = len(lines)
        
        iteration = 0
        
        while start < lines_len:
            iteration += 1
            if iteration > 1000:  # Safeguard against infinite loops
                break
                
            end = min(start + chunk_size, lines_len)
            chunk = ''.join(lines[start:end])
            chunks.append(chunk)
            # Store the previous start position to detect lack of progress
            prev_start = start
            start = end - chunk_overlap
            
            # Handle negative or unchanged start value to prevent infinite loops
            if start < 0 or (start >= end and start < lines_len) or start == prev_start:
                break
        
        return chunks
