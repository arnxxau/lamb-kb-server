"""
Query service for retrieving data from collections.

This module provides services for querying collections using query plugins.
"""

import time
from typing import Dict, List, Any, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from plugins.base import PluginRegistry, QueryPlugin


class QueryService:
    """Service for querying collections."""
    
    @classmethod
    def get_plugin(cls, name: str) -> Optional[QueryPlugin]:
        """Get a query plugin by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        plugin_class = PluginRegistry.get_query_plugin(name)
        if plugin_class:
            return plugin_class()
        return None
    
    @classmethod
    def list_plugins(cls) -> List[Dict[str, Any]]:
        """List all available query plugins.
        
        Returns:
            List of plugins with metadata
        """
        return PluginRegistry.list_query_plugins()
    
    @classmethod
    def query_collection(
        cls, 
        db: Session,
        collection_id: int,
        query_text: str,
        plugin_name: str,
        plugin_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query a collection using the specified plugin.
        
        Args:
            db: Database session
            collection_id: ID of the collection
            query_text: Query text
            plugin_name: Name of the plugin to use
            plugin_params: Parameters for the plugin
            
        Returns:
            Query results with timing information
            
        Raises:
            HTTPException: If the plugin is not found or query fails
        """
        plugin = cls.get_plugin(plugin_name)
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail=f"Query plugin '{plugin_name}' not found"
            )
        
        try:
            # Add db to plugin params
            params = plugin_params.copy()
            params["db"] = db
            
            # Record start time
            start_time = time.time()
            
            # Execute query
            results = plugin.query(
                collection_id=collection_id,
                query_text=query_text,
                **params
            )
            
            # Record end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            return {
                "results": results,
                "count": len(results),
                "timing": {
                    "total_seconds": elapsed_time,
                    "total_ms": elapsed_time * 1000  # Add milliseconds for test script
                },
                "query": query_text
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to query collection: {str(e)}"
            )
