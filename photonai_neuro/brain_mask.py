from typing import Union

import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
from sklearn.base import BaseEstimator

from photonai.photonlogger.logger import logger

from photonai_neuro.objects import RoiObject, NiftiConverter
from photonai_neuro.atlas_library import AtlasLibrary


class BrainMask(BaseEstimator):
    """BrainMask

    Transform based on given mask image.
    """

    def __init__(self, mask_image: Union[str, RoiObject, None] = 'MNI_ICBM152_WholeBrain',
                 affine: Union[np.ndarray, None] = None, shape: Union[tuple, list, None] = None,
                 mask_threshold: Union[float, None] = 0.5, extract_mode: str = 'vec'):
        self.mask_image = mask_image

        self.affine = affine
        self.shape = shape
        self.masker = None
        self.extract_mode = extract_mode
        self.mask_threshold = mask_threshold

    def fit(self, X, y):
        return self

    def transform(self, X, y=None, **kwargs) -> np.ndarray:

        if self.affine is None and self.shape is None:
            self.affine, self.shape = NiftiConverter.get_format_info_from_first_image(X)

        # checking ground mask
        if isinstance(self.mask_image, str):
            local_mask_image = AtlasLibrary().get_mask(self.mask_image, self.affine, self.shape, self.mask_threshold)
        elif isinstance(self.mask_image, RoiObject):
            local_mask_image = self.mask_image
        else:
            msg = "Expected mask_image as instance of str or RoiObject, found {}".format(type(self.mask_image))
            logger.error(msg)
            raise TypeError(msg)

        if not local_mask_image.is_empty:
            self.masker = NiftiMasker(mask_img=local_mask_image.mask, target_affine=self.affine,
                                      target_shape=self.shape, dtype='float32')
            try:
                single_roi = self.masker.fit_transform(X)
            except BaseException as e:
                logger.error(e)
                single_roi = None

            assert self._check_single_roi(X, single_roi)

            if self.extract_mode == 'vec':
                return np.asarray(single_roi)
            elif self.extract_mode == 'mean':
                return np.mean(single_roi, axis=1)
            elif self.extract_mode == 'box':
                return BrainMask._get_box(X, local_mask_image)
            elif self.extract_mode == 'img':
                return self.masker.inverse_transform(single_roi)
            else:
                msg = "Currently there are no other methods than 'vec', 'mean', 'img' and 'box'!"
                logger.error(msg)
                raise NameError(msg)
        else:
            msg = "Skipping self.mask_image " + self.mask_image.label + " because it is empty."
            logger.error(msg)
            raise ValueError(msg)

    def inverse_transform(self, X, y=None, **kwargs):
        """
        Inverse transformation. Only available and well-defined for extract_mode == 'vec'.

        """
        if not self.extract_mode == 'vec':
            msg = "BrainMask extract_mode={} is not supported with inverse_transform".format(self.extract_mode)
            logger.error(msg)
            raise NotImplementedError(msg)

        return self.masker.inverse_transform(X)

    @staticmethod
    def _check_single_roi(X, single_roi):
        if single_roi is not None:
            return True
        else:
            if isinstance(X, str):
                msg = "Extracting ROI failed for " + X
            elif isinstance(X, list) and isinstance(X[0], str):
                msg = "Extracting ROI failed for item in" + str(X)
            else:
                msg = "Extracting ROI failed for nifti image obj. Cannot trace back path of failed file."
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _get_box(in_imgs, roi):
        # get ROI infos
        map = roi.mask.get_fdata()
        true_points = np.argwhere(map)
        corner1 = true_points.min(axis=0)
        corner2 = true_points.max(axis=0)
        box = []
        for img in in_imgs:
            if isinstance(img, str):
                data = image.load_img(img).get_fdata()
            else:
                data = img.get_fdata()
            tmp = data[corner1[0]:corner2[0] + 1, corner1[1]:corner2[1] + 1, corner1[2]:corner2[2] + 1]
            box.append(tmp)
        return np.asarray(box)
