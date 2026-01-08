"""
Package Evaluation pour le projet RDD2022
Contient les outils d'évaluation et de comparaison des modèles
"""

from .metrics import (
    evaluate_model_on_dataset,
    calculate_confusion_matrix,
    calculate_pixel_accuracy,
    calculate_class_metrics,
    calculate_metrics_per_class
)

from .visualization import (
    plot_metrics_comparison,
    plot_per_class_metrics,
    plot_confusion_matrices,
    plot_predictions_grid,
    plot_training_curves,
    create_evaluation_report
)

from .evaluate import (
    evaluate_all_models,
    compare_models,
    save_evaluation_results
)

__all__ = [
    # Metrics
    'evaluate_model_on_dataset',
    'calculate_confusion_matrix',
    'calculate_pixel_accuracy',
    'calculate_class_metrics',
    'calculate_metrics_per_class',
    
    # Visualization
    'plot_metrics_comparison',
    'plot_per_class_metrics',
    'plot_confusion_matrices',
    'plot_predictions_grid',
    'plot_training_curves',
    'create_evaluation_report',
    
    # Main evaluation
    'evaluate_all_models',
    'compare_models',
    'save_evaluation_results'
]
