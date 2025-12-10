"""
Metrics utilities for evaluation.
"""
import numpy as np
from typing import Dict
from app.utils.logger import get_logger

logger = get_logger(__name__)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient for segmentation.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def calculate_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_macro': float(f1)
    }
