from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from typing import Union

from nilearn import image
from nibabel.nifti1 import Nifti1Image

from photonai.photonlogger.logger import logger


class NiftiConverter:
    """
    Handle transformation for different inputs to homogeneous output.
    Output is a Nifti1Image object.
    """

    @classmethod
    def transform(cls, X):
        n_subjects = 1
        load_data = None
        msg = None
        # load all niftis to memory
        if isinstance(X, list) or isinstance(X, np.ndarray):
            n_subjects = len(X)
            if all([isinstance(x, str) for x in X]):
                load_data = image.load_img(X)
            elif all([isinstance(x, np.ndarray) for x in X]):
                n_subjects = X.shape[0]
                load_data = image.load_img(X)
            elif all([isinstance(x, Nifti1Image) for x in X]):
                load_data = image.load_img(X)
            else:
                msg = "Cannot interpret the types in the given input_data list."
        elif isinstance(X, str):
            load_data = image.load_img(X)
        elif isinstance(X, Nifti1Image):
            load_data = X
        else:
            msg = "Can only process strings as file paths to nifti images or nifti image object"
        if msg:
            logger.error(msg)
            raise ValueError(msg)

        return load_data, n_subjects


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
