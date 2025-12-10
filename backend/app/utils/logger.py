"""
Logger utility for getting configured logger instances.
"""
import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name or __name__)
