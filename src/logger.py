import logging
import sys
from pathlib import Path

# Fix relative import issue if run directly or as module
try:
    from src.config import LOG_FILE_PATH
except ImportError:
    # Fallback to local import or absolute path if not run as a package
    LOG_FILE_PATH = Path(__file__).resolve().parent.parent / "logs" / "app.log"

def setup_logger(name: str = "elevator_predictive_maintenance") -> logging.Logger:
    """
    Sets up a logger that outputs to both stdout and a log file.
    
    Args:
        name: Name of the logger.
        
    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If the logger is already configured, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create formats
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
