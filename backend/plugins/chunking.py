"""
Chunking utility functions for text content processing.

This module provides reusable functions for splitting content into chunks
based on different units (characters, words, lines) with configurable
chunk sizes and overlaps.
"""

import re
from enum import Enum
from typing import List


class ChunkUnit(Enum):
    """Enumeration of possible chunking units."""
    CHAR = "char"
    WORD = "word"
    LINE = "line"
    
    def __str__(self):
        """Return the value of the enum as string."""
        return self.value


def split_content(content: str, chunk_size: int, 
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
    if chunk_unit == ChunkUnit.CHAR:
        return split_by_chars(content, chunk_size, chunk_overlap)
    elif chunk_unit == ChunkUnit.WORD:
        return split_by_words(content, chunk_size, chunk_overlap)
    elif chunk_unit == ChunkUnit.LINE:
        return split_by_lines(content, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unsupported chunk unit: {chunk_unit}")


def split_by_chars(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
    
    while start < content_len:
        end = min(start + chunk_size, content_len)
        chunk = content[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        
        # Safety check to prevent infinite loops
        if start >= end:
            break
    
    return chunks


def split_by_words(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
    
    while start < words_len:
        end = min(start + chunk_size, words_len)
        chunk = ''.join(words[start:end])
        chunks.append(chunk)
        
        # Move start position for next chunk, with overlap
        start = end - chunk_overlap
        
        # Safety check to prevent infinite loops
        if start >= end or start < 0:
            break
    
    return chunks


def split_by_lines(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
    
    while start < lines_len:
        end = min(start + chunk_size, lines_len)
        chunk = ''.join(lines[start:end])
        chunks.append(chunk)
        
        # Move start position for next chunk, with overlap
        start = end - chunk_overlap
        
        # Safety check to prevent infinite loops
        if start >= end or start < 0:
            break
    
    return chunks