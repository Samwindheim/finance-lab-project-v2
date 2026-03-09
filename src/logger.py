import logging
import sys
import os

# Add project root to the Python path to allow root-level imports like 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and cleaner layout."""
    
    # ANSI Color Codes
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    FORMAT = "  %(levelname)-7s | %(message)s"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record):
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        
        # Color the level name
        levelname = record.levelname
        record.levelname = f"{log_color}{levelname}{self.RESET}"
        
        # Format the actual message
        result = super().format(record)
        
        # Restore levelname for other handlers if any
        record.levelname = levelname
        return result

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
        formatter = CustomFormatter(config.LOG_FORMAT)
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
    return logger

