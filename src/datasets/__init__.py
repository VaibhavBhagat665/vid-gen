"""
Dataset registry and factory
"""

from .hdvila import HDVILADataset, collate_fn
from .webvid import WebVidDataset
from .ucf101 import UCF101Dataset


DATASETS = {
    'hdvila': HDVILADataset,
    'webvid': WebVidDataset,
    'ucf101': UCF101Dataset,
}


def get_dataset(dataset_name, **kwargs):
    """
    Factory function to get dataset by name.
    
    Args:
        dataset_name: Name of dataset ('hdvila' or 'webvid')
        **kwargs: Arguments to pass to dataset constructor
        
    Returns:
        Dataset instance
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    return DATASETS[dataset_name](**kwargs)


__all__ = ['HDVILADataset', 'WebVidDataset', 'UCF101Dataset', 'get_dataset', 'collate_fn']
