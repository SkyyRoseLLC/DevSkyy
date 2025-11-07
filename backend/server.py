#!/usr/bin/env python3
"""
Backend server entry point for The Skyy Rose Collection Platform
This file imports and runs the main FastAPI application.
"""

import os
import sys
import logging

import uvicorn
from main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path so we can import from main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from main.py

# This allows uvicorn to find the app when running with "uvicorn server:app"
if __name__ == "__main__":

    logger.info("🚀 Starting DevSkyy Enhanced Backend Server")
    logger.info("🌟 Brand Intelligence: MAXIMUM")
    logger.info("📚 Continuous Learning: ACTIVE")
    logger.info("⚡ Setting the Bar for AI Agents")
    logger.info("🌐 Backend server starting on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
