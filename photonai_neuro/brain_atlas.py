from nilearn import image, masking, _utils
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import time
from typing import Union

from photonai.photonlogger.logger import logger
from photonai_neuro.atlas_library import AtlasLibrary
from photonai_neuro.objects import NiftiConverter


class BrainAtlas(BaseEstimator, TransformerMixin):
    """
    BrainAtlas is a transformer calculate brain atlases from input niftis.

    ToDo
        + check RAS vs. LPS view-type and provide warning
        - unit tests
        Later
        - add support for overlapping ROIs and probabilistic atlases using 4d-nii
        - add support for 4d resting-state data using nilearn
    """
    def __init__(self, atlas_name: str, extract_mode: str = 'vec', mask_threshold: float = None,
                 background_id: int = 0, rois: Union[list, str] = 'all'):
        """
        Initialize the object.

        Parameters:
            atlas_name:
                Name of specific ytlas. Possible values can be looked up in the AtlasLibrary.

            extract_mode:
                The mode performing on ROI. Possible values: ['vec', 'mean'].

            mask_threshold:
                Mask Threshold. value < mask_threshold => value = 0.

            background_id:
                The background ID for ROI.
        """

        self._extract_mode = None

        self.extract_mode = extract_mode
        self.atlas_name = atlas_name
        # collection mode default to concat --> can only be overwritten by AtlasMapper
        self.collection_mode = 'concat'
        self.mask_threshold = mask_threshold
        self.background_id = background_id
        self.rois = rois
        self.box_shape = []
        self.is_transformer = True
        self.mask_indices = None
        self.affine = None
        self.shape = None
        self.needs_y = False
        self.needs_covariates = False
        self.roi_allocation = {}

    @property
    def extract_mode(self):
        return self._extract_mode

    @extract_mode.setter
    def extract_mode(self, value):
        if value in ["vec", "mean"]:
            self._extract_mode = value
        else:
            msg = "Currently there are no other methods than 'vec', 'mean', 'img' and 'box' supported!"
            logger.error(msg)
            raise ValueError(msg)

    def fit(self, X, y):
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Transform X by applying the underlying atlas.

        Parameters:
            X: input data
            y: targets
            **kwargs: -

        Returns:
            Data for each ROI for given brain atlas in concat or list form.

        """
        X, n_subjects = NiftiConverter.transform(X)

        if self.collection_mode == 'list' or self.collection_mode == 'concat':
            collection_mode = self.collection_mode
        else:
            msg = "Collection mode {} not supported. Use 'list' or 'concat' instead." \
                  "Falling back to concat mode.".format(self.collection_mode)
            logger.error(msg)
            raise ValueError(msg)

        # 1. validate if all X are in the same space and have the same voxelsize and have the same orientation

        # get ROI mask
        self.affine, self.shape = NiftiConverter.get_format_info_from_first_image(X)
        atlas_obj = AtlasLibrary().get_atlas(self.atlas_name, self.affine, self.shape, self.mask_threshold)
        roi_objects = self._get_rois(atlas_obj, which_rois=self.rois, background_id=self.background_id)

        roi_data = np.array([], dtype=np.int64).reshape(n_subjects, 0) \
            if self.collection_mode == "concat" else [list() for _ in range(n_subjects)]
        t1 = time.time()

        # convert to series and C ordering since this will speed up the masking process
        series = _utils.as_ndarray(_utils.niimg._safe_get_data(X), dtype='float32', order="C", copy=True)
        mask_indices = np.array([]) if self.collection_mode == "concat" else list()

        # calculate roi_data for every ROI object by looping
        # ToDo: Performance
        for i, roi in enumerate(roi_objects):
            self.roi_allocation[roi.label] = i

            logger.debug("Extracting ROI {}.".format(roi.label))
            # simply call apply_mask to extract one roi
            extraction = self.apply_mask(series, roi.mask)

            if self.extract_mode == 'mean':
                extraction = np.reshape(np.mean(extraction, axis=1), (-1, 1))
            if n_subjects == 1:
                extraction = np.reshape(extraction, (1, -1))

            if collection_mode == 'list':
                for sub_i in range(extraction.shape[0]):
                    roi_data[sub_i].append(extraction[sub_i])
                mask_indices.append(i)
            else:
                roi_data = np.concatenate([roi_data, extraction], axis=1)
                mask_indices = np.concatenate([mask_indices, np.ones(extraction[0].size) * i])

        self.mask_indices = mask_indices

        elapsed_time = time.time() - t1
        logger.debug("Time for extracting {} ROIs in {} subjects: {} seconds".format(len(roi_objects),
                                                                                     n_subjects, elapsed_time))
        return roi_data

    def apply_mask(self, series, mask_img):
        """
        Apply mask on series.
        :param series: np.ndarray, data working on
        :param mask_img:
        :return:
        """
        mask_img = _utils.check_niimg_3d(mask_img)

        # Todo: consider extraction mode!
        return series[mask_img.dataobj.astype(bool)].T

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Reconstruct image from transformed data.

        :param X: data
        :param y: targets
        :param kwargs:
        :return:
        """
        X = np.asarray(X)

        # get ROI masks
        atlas_obj = AtlasLibrary().get_atlas(self.atlas_name, self.affine, self.shape, self.mask_threshold)
        roi_objects = self._get_rois(atlas_obj, which_rois=self.rois, background_id=self.background_id)

        unmasked = np.squeeze(np.zeros_like(atlas_obj.map, dtype='float32'))

        for i, roi in enumerate(roi_objects):
            mask, mask_affine = masking._load_mask_img(roi.mask)
            mask_img = image.new_img_like(roi.mask, mask, mask_affine)
            mask_data = _utils.as_ndarray(mask_img.get_fdata(), dtype=np.bool)

            if self.collection_mode == 'list':
                unmasked[mask_data] = X[i]
            else:
                unmasked[mask_data] = X[self.mask_indices.reshape(X.shape) == i]

        new_image = image.new_img_like(atlas_obj.atlas, unmasked)
        return new_image

    def _validity_check_roi_extraction(self, X, y=None, filename='validity_check.nii', **kwargs):
        new_image = self.inverse_transform(X, y, **kwargs)
        new_image.to_filename(filename)

    @staticmethod
    def _get_rois(atlas_obj, which_rois='all', background_id=0):

        if isinstance(which_rois, str):
            if which_rois == 'all':
                return [roi for roi in atlas_obj.roi_list if roi.index != background_id]
            else:
                return AtlasLibrary().find_rois_by_label(atlas_obj, [which_rois])

        elif isinstance(which_rois, int):
            return AtlasLibrary().find_rois_by_index(atlas_obj, [which_rois])

        elif isinstance(which_rois, list):
            if isinstance(which_rois[0], str):
                if which_rois[0].lower() == 'all':
                    return [roi for roi in atlas_obj.roi_list if roi.index != background_id]
                else:
                    return AtlasLibrary().find_rois_by_label(atlas_obj, which_rois)
            else:
                return AtlasLibrary().find_rois_by_index(atlas_obj, which_rois)
