from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from typing import Union


class RoiObject:

    def __init__(self, index=0, label='', size=None, mask=None):
        self.index = index
        self.label = label
        self.size = size
        self.mask = mask
        self.is_empty = False


class NeuroTransformerMixin:

    def __init__(self, output_img: bool = False):
        self.output_img = output_img


class RoiFilter(BaseEstimator, TransformerMixin):

    def __init__(self, roi_allocation: dict, mask_indices: Union[np.ndarray, list], rois = []):
        self.roi_allocation = roi_allocation
        self.mask_indices = mask_indices
        self._rois = None
        self.rois = rois

    @property
    def rois(self):
        return self._rois

    @rois.setter
    def rois(self, value):
        if isinstance(value, str):
            self._rois = [value]
        else:
            self._rois = value
        self.rois_indices = [self.roi_allocation[roi] for roi in self.rois]
        self.filter_indices = np.array([True if x in self.rois_indices else False for x in self.mask_indices])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **kwargs):
        return_data = X[:, self.filter_indices]
        return return_data


class MaskObject:

    def __init__(self, name: str = '', mask_file: str = '', mask=None):
        self.name = name
        self.mask_file = mask_file
        self.mask = mask
        self.is_empty = False


class AtlasObject:

    def __init__(self, name='', path='', labels_file='', mask_threshold=None, affine=None, shape=None, indices=list()):
        self.name = name
        self.path = path
        self.labels_file = labels_file
        self.mask_threshold = mask_threshold
        self.indices = indices
        self.roi_list = list()
        self._rois_available = None
        self.rois_active = []
        self.rois_allocation = None
        self.map = None
        self.atlas = None
        self.affine = affine
        self.shape = shape

        self.rois_available = []
