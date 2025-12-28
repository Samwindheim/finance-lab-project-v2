import logging
import sys
import os

# Add project root to the Python path to allow root-level imports like 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def setup_logger(name: str):
    """Sets up a logger with the specified name and configuration."""
    logger = logging.getLogger(name)
    
    # Only add handlers if the logger doesn't already have them
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter(config.LOG_FORMAT)
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
    return logger

