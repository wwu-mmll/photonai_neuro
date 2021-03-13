import numpy as np
from typing import Union, List
import warnings

from sklearn.base import BaseEstimator
from nilearn.image import resample_img, smooth_img, index_img, load_img
from nibabel.nifti1 import Nifti1Image
from skimage.util.shape import view_as_windows

from photonai.photonlogger.logger import logger

from photonai_neuro.objects import NeuroTransformerMixin


class SmoothImages(BaseEstimator, NeuroTransformerMixin):
    """PipelineElement to perform nilearns smooth_img function."""

    def __init__(self, fwhm: Union[int, List, str] = 2, output_img: bool = False):
        """
        Initialize the object.

        Parameters:
            fwhm:
                Smoothing strength, as a Full-Width at Half Maximum, in millimeters.
                If a scalar is given, width is identical on all three directions.
                A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
                If fwhm == ‘fast’, a fast smoothing will be performed with a filter [0.2, 1, 0.2]
                in each direction and a normalisation to preserve the scale.
                If fwhm is None, no filtering is performed (useful when just removal of non-finite values is needed).
                * cited from nilearn: https://nilearn.github.io/modules/generated/nilearn.image.smooth_img.html

            output_img:
                Indicates the output format. False -> array,  True -> object (Nifti1Image).

        """
        super(SmoothImages, self).__init__(output_img=output_img)

        self._fwhm = None
        self.fwhm = fwhm

    def fit(self, X, y=None, **kwargs):
        return self

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        if isinstance(fwhm, (int, float)):  # allowing float to improve optimization
            self._fwhm = [fwhm, fwhm, fwhm]
        elif isinstance(fwhm, list) and len(fwhm) == 3 and all(isinstance(x, (int, float)) for x in fwhm):
            self._fwhm = fwhm
        elif fwhm == 'fast':
            self._fwhm = fwhm
        elif fwhm is None:
            self._fwhm = None
            warn_msg = "The fwhm in SmoothImages is None, no filtering is performed (useful when just " \
                       "removal of non-finite values is needed). "
            logger.warning(warn_msg)
            warnings.warn(warn_msg)
        else:
            msg = "SmoothImages expected fwhm as int, as str=='fast' or a list of three ints like [3, 3, 3]."
            logger.error(msg)
            raise ValueError(msg)

    def transform(self, X, y=None, **kwargs):

        if isinstance(X, list) and len(X) == 1:
            smoothed_img = smooth_img(X[0], fwhm=self.fwhm)
        else:
            smoothed_img = smooth_img(X, fwhm=self.fwhm)

        if not self.output_img:
            if isinstance(smoothed_img, list):
                smoothed_img = np.asarray([img.dataobj for img in smoothed_img])
            else:
                return smoothed_img.dataobj
        return smoothed_img


class ResampleImages(BaseEstimator, NeuroTransformerMixin):
    """
     Resampling voxel size based on nilearns resample_img function.
     This object creates the target_affine = np.diag(voxel_size) as 3x3 matrix.

    """
    def __init__(self, voxel_size: Union[float, int, List] = 3,
                 interpolation: str = 'nearest', output_img: bool = False):
        """
        Initialize the object.

        Parameters:
            voxel_size:
                Value to create target_affine matrix for resmapled_img function.
                Length of list has to be in [3, 4].

            interpolation:
                Set the resample method.

            output_img:
                Indicates the output format. False -> np.ndarray,  True -> object (Nifti1Image).

        """
        super(ResampleImages, self).__init__(output_img=output_img)
        self._voxel_size = None
        self.voxel_size = voxel_size
        self._interpolation = None
        self.interpolation = interpolation

    def fit(self, X: np.ndarray, y: Union[None, np.ndarray] = None):
        """
        Required function without any effect.

        Parameters:
            X:
                The input samples as Niimg-like object of shape [n_samples, 1].

            y:
                The input targets of shape [n_samples, 1].

        Returns:
            IMPORTANT, must return self!

        """
        return self

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value):
        if value in ['continuous', 'linear', 'nearest']:
            self._interpolation = value
        else:
            msg = "Got unexpected interpolation. Please use one of ['continuous', 'linear' 'nearest']."
            logger.error(msg)
            raise NameError(msg)

    @property
    def voxel_size(self):
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size):
        if isinstance(voxel_size, (int, float)):
            self._voxel_size = [voxel_size, voxel_size, voxel_size]
        elif isinstance(voxel_size, list) and len(voxel_size) in [3, 4] \
                and all(isinstance(x, (int, np.int, np.int64, np.int32, float)) for x in voxel_size):
            self._voxel_size = voxel_size
        else:
            msg = "ResampleImages expected voxel_size as int or a list of three/four ints like [3, 3, 3]."
            logger.error(msg)
            raise ValueError(msg)

    def transform(self, X, y=None, **kwargs):
        target_affine = np.diag(self.voxel_size)

        if isinstance(X, list) and len(X) == 1:
            resampled_img = resample_img(X[0], target_affine=target_affine, interpolation=self.interpolation)
        else:
            resampled_img = resample_img(X, target_affine=target_affine, interpolation=self.interpolation)

        if self.output_img:
            if len(resampled_img.shape) == 3:
                if isinstance(resampled_img, (list, np.ndarray)):
                    return resampled_img
                else:
                    return [resampled_img]
            else:
                resampled_img = [index_img(resampled_img, i) for i in range(resampled_img.shape[-1])]
        else:
            if len(resampled_img.shape) == 3:
                return resampled_img.dataobj
            else:
                resampled_img = np.moveaxis(resampled_img.dataobj, -1, 0)

        return resampled_img


class PatchImages(BaseEstimator, NeuroTransformerMixin):
    """
    ToDo: Check correct input /output!
    """

    def __init__(self, patch_size=25, nr_of_processes=1):
        super(PatchImages, self).__init__(output_img=True)
        # Todo: give cache folder to mother class

        self.nr_of_processes = nr_of_processes
        self.patch_size = patch_size

        msg = "Use PatchImages wisely: not tested in content."
        logger.warning(msg)
        warnings.warn(msg)

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        logger.info("Drawing patches")
        return self.draw_patches(X, self.patch_size)

    @staticmethod
    def draw_patches(patch_x, patch_size):

        if isinstance(patch_x, str):
            patch_x = np.ascontiguousarray(load_img(patch_x).get_data())
        elif isinstance(patch_x, Nifti1Image):
            patch_x = np.ascontiguousarray(patch_x.dataobj)
        elif isinstance(patch_x, np.ndarray):
            if len(patch_x.shape) == 1:
                patch_x = list(patch_x)
        elif isinstance(patch_x, list):
            pass
        else:
            msg = "Could not read input data."
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(patch_x, list):
            if all([isinstance(px, str) for px in patch_x]):
                patch_x = [np.ascontiguousarray(load_img(px).get_data()) for px in patch_x]
            return_list = []
            for p in patch_x:

                return_list.append(PatchImages.draw_patch_from_mri(p, patch_size))
            return return_list

        return PatchImages.draw_patch_from_mri(patch_x, patch_size)

    @staticmethod
    def draw_patch_from_mri(patch_x: np.ndarray, patch_size):

        patches_drawn = view_as_windows(patch_x, (patch_size, patch_size, 1), step=1)

        patch_list_length = patches_drawn.shape[0]
        patch_list_width = patches_drawn.shape[1]

        output_matrix = patches_drawn[0:patch_list_length:patch_size, 0:patch_list_width:patch_size, :, :]

        # TODO: Reshape First 3 Matrix Dimensions into 1, which will give 900 images
        output_matrix = output_matrix.reshape((-1,
                                               output_matrix.shape[3],
                                               output_matrix.shape[4],
                                               output_matrix.shape[5]))
        output_matrix = np.squeeze(output_matrix)

        return output_matrix

    def copy_me(self):
        return PatchImages(self.patch_size, self.nr_of_processes)

    def _draw_single_patch(self):
        raise NotImplementedError("Not implemented yet.")
