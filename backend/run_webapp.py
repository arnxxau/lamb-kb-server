#!/usr/bin/env python3
"""
Run script for the Lamb Knowledge Base Web Application
"""

import os
from pathlib import Path
from lamb_kb_webapp import app

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Get host and port from environment variables or use defaults
    host = os.getenv("LAMB_WEBAPP_HOST", "0.0.0.0")
    port = int(os.getenv("LAMB_WEBAPP_PORT", "8083"))
    
    print(f"Starting Lamb KB Web Explorer on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask application
    app.run(host=host, port=port, debug=True) 