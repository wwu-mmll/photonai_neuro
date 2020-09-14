import numpy as np

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
        elif isinstance(X, str) or (isinstance(X, np.ndarray) and len(X.shape) in [3, 4]):
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
