"""
Logging configuration for BTSC-UNet-ViT project.
Provides structured logging with context fields throughout the application.
"""
import logging
import logging.config
from typing import Any, Dict


class ContextFilter(logging.Filter):
    """Add context fields to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add default context fields if not present
        if not hasattr(record, 'image_id'):
            record.image_id = None
        if not hasattr(record, 'path'):
            record.path = None
        if not hasattr(record, 'stage'):
            record.stage = None
        return True


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure project-wide logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'filters': {
            'context_filter': {
                '()': ContextFilter,
            }
        },
        'formatters': {
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s | context=%(image_id)s,%(path)s,%(stage)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s | %(levelname)-8s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filters': ['context_filter'],
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filters': ['context_filter'],
                'filename': 'backend/app/resources/app.log',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False
            },
            'fastapi': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized", extra={
        'image_id': None,
        'path': None,
        'stage': 'initialization'
    })
