from nibabel.nifti1 import Nifti1Image
from nilearn import image
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Tuple

from photonai.photonlogger.logger import logger


class NiftiConverter:
    """
    Handle transformation for different inputs to homogeneous output.
    Output is a Nifti1Image object and the count of subjects (4th dimension).

    """
    @classmethod
    def transform(cls, X: Union[list, np.ndarray, Nifti1Image, str]) -> Tuple[Nifti1Image, int]:
        n_subjects = 1
        load_data = None
        msg = None
        # load all niftis to memory
        if isinstance(X, list) or isinstance(X, np.ndarray):
            n_subjects = len(X)
            if n_subjects and all([isinstance(x, str) for x in X]):
                load_data = image.load_img(X)
            elif n_subjects and all([isinstance(x, np.ndarray) for x in X]):
                n_subjects = X.shape[0]
                load_data = image.load_img(X)
            elif n_subjects and all([isinstance(x, Nifti1Image) for x in X]):
                load_data = image.load_img(X)
            else:
                msg = "Cannot interpret the types in the given input_data list."
        elif isinstance(X, str) or (isinstance(X, np.ndarray) and len(X.shape) in [3, 4]):
            load_data = image.load_img(X)
        elif isinstance(X, Nifti1Image):
            load_data = X
        else:
            msg = "Can only process strings as file paths to nifti images or nifti image objects."
        if msg:
            logger.error(msg)
            raise ValueError(msg)

        return load_data, n_subjects

    @staticmethod
    def get_format_info_from_first_image(X: Union[list, np.ndarray, Nifti1Image, str]) \
            -> Tuple[np.ndarray, Union[tuple, list]]:
        """
        Return Affine and Shape of first image.

        Parameter:
            X:
                Input data - return only affine and shape of first image.

        """
        img, n_subjects = NiftiConverter.transform(X)
        if n_subjects > 1:
            img = img.slicer[:, :, :, 0]

        if img is not None:
            return img.affine, img.shape[:3]
        else:
            msg = "Could not load image for affine and shape definition."
            logger.error(msg)
            raise ValueError(msg)


class RoiObject:
    """
    The RoiObjects guarantees a clear assignment of ROIs within the mask.
    The ROI known as `label` with size `size` defined by the index in the mask.
    Every ROI got an unique index stored in an mask object.

    For later versions: Performance advantage if all ROI objects save references of mask
    and the property roi_mask returns the applied index on mask.

    """
    def __init__(self, index: int, label: str = '', size: int = None, mask=None):
        """
        Inizialize the object.

        Parameters:
            index:
                Unique index for label and mask.

            label:
                Label of region of interest (ROI).

            size:
                Size of positiv voxels in mask.

            mask:
                Mask containing ROI indeces.
        """
        self.index = index
        self.label = label
        self.mask = mask
        self.size = size
        self.is_empty = True if self.size == 0 else False


class NeuroTransformerMixin(TransformerMixin):
    """Abstract class for objects with possible output NiftiImage/np.ndarray."""

    def __init__(self, output_img: bool = False):
        """
        Initialize the object.

        Parameters:
            output_img:
                True -> return is instance of NiftiImage,
                False -> return is instance of np.ndarray (nii.dataobj).

        """
        self.output_img = output_img


class MaskObject:

    def __init__(self, name: str = '', mask_file: str = '', mask=None):
        self.name = name
        self.mask_file = mask_file
        self.mask = mask
        self.is_empty = False


class AtlasObject:

    def __init__(self,
                 name: str = '',
                 path: str = '',
                 labels_file: str = '',
                 mask_threshold: float = None,
                 affine: np.ndarray = None,
                 shape: Union[tuple, np.ndarray] = None,
                 indices: list = None):

        self._rois_available = None

        self.name = name
        self.path = path
        self.labels_file = labels_file
        self.mask_threshold = mask_threshold
        self.indices = [] if indices is None else indices
        self.roi_list = []
        self.rois_active = []
        self.rois_allocation = None
        self.map = None
        self.atlas = None
        self.affine = affine
        self.shape = shape
        self.rois_available = []
