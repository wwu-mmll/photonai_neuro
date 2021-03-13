import glob
import inspect
import warnings
from os import path
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

from photonai.photonlogger.logger import logger

from photonai_neuro.objects import MaskObject, AtlasObject, RoiObject


class AtlasLibrary:
    """
    The AtlasLibrary manages access to Atlases on hard drive and memory.

    Every access follows the scheme:
        1. load into RAM to dictonary LIBRARY with (name, affine, shape, threshold) as key
            and utils.AtlasObject as value.
        2. Provide entrance to LIBRARY via get_atlas.

    """
    ATLAS_DICTIONARY = {'AAL': 'AAL.nii.gz',
                        'HarvardOxford_Cortical_Threshold_25': 'HarvardOxford-cort-maxprob-thr25.nii.gz',
                        'HarvardOxford_Subcortical_Threshold_25': 'HarvardOxford-sub-maxprob-thr25.nii.gz',
                        'HarvardOxford_Cortical_Threshold_50': 'HarvardOxford-cort-maxprob-thr50.nii.gz',
                        'HarvardOxford_Subcortical_Threshold_50': 'HarvardOxford-sub-maxprob-thr50.nii.gz',
                        'Yeo_7': 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz',
                        'Yeo_7_Liberal': 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz',
                        'Yeo_17': 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz',
                        'Yeo_17_Liberal': 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz',
                        'Schaefer2018_100Parcels_7Networks': 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_200Parcels_7Networks': 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_300Parcels_7Networks': 'Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_400Parcels_7Networks': 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_500Parcels_7Networks': 'Schaefer2018_500Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_600Parcels_7Networks': 'Schaefer2018_600Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_700Parcels_7Networks': 'Schaefer2018_700Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_800Parcels_7Networks': 'Schaefer2018_800Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_900Parcels_7Networks': 'Schaefer2018_900Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_1000Parcels_7Networks': 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_100Parcels_17Networks': 'Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_200Parcels_17Networks': 'Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_300Parcels_17Networks': 'Schaefer2018_300Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_400Parcels_17Networks': 'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_500Parcels_17Networks': 'Schaefer2018_500Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_600Parcels_17Networks': 'Schaefer2018_600Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_700Parcels_17Networks': 'Schaefer2018_700Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_800Parcels_17Networks': 'Schaefer2018_800Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_900Parcels_17Networks': 'Schaefer2018_900Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',
                        'Schaefer2018_1000Parcels_17Networks': 'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm.nii.gz',}

    MASK_DICTIONARY = {'MNI_ICBM152_GrayMatter': 'mni_icbm152_gm_tal_nlin_sym_09a.nii.gz',
                       'MNI_ICBM152_WhiteMatter': 'mni_icbm152_wm_tal_nlin_sym_09a.nii.gz',
                       'MNI_ICBM152_WholeBrain': 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz',
                       'Cerebellum': 'P_08_Cere.nii.gz'}

    LIBRARY = dict()  # all active atlases

    def __init__(self):
        self.photon_atlases = self._load_photon_atlases()
        self.photon_masks = self._load_photon_masks()

    def get_atlas(self,
                  atlas_name: str,
                  target_affine: Union[np.ndarray, None] = None,
                  target_shape: Union[list, tuple, None] = None,
                  mask_threshold: Union[float, None] = None) -> AtlasObject:
        """
        Read access to LIBRARY. Search for key (name, affine, shape, threshold).
        If key is set, return the suitable value, else add/return new AtlasObject to/from LIBRARY.

        Parameters:
            atlas_name:
                Name of atlas.

            target_affine:
                If specified, the atlas-image is resampled corresponding to this new affine.
                target_affine can be a 3x3 or a 4x4 matrix.

            target_shape:
                If specified, the atlas-image will be resized to match this new shape.
                len(target_shape) must be equal to 3.
                If target_shape is specified, a target_affine of shape (4, 4) must also be given.

            mask_threshold:
                Threshold for defining background.

        """
        if (atlas_name, str(target_affine), str(target_shape), str(mask_threshold)) not in \
                list(AtlasLibrary.LIBRARY.keys()):
            self._add_atlas_to_library(atlas_name, target_affine, target_shape, mask_threshold)

        return AtlasLibrary.LIBRARY[(atlas_name, str(target_affine), str(target_shape), str(mask_threshold))]

    def get_mask(self,
                 mask_name: str,
                 target_affine: Union[np.ndarray, None] = None,
                 target_shape: Union[tuple, list, None] = None,
                 mask_threshold: float = 0.5):
        """
        Mask equivalent to get_atlas. Add to LIBRARY if not exists and returns.

        Parameters:
            mask_name:
                Name of mask.

            target_affine:
                If specified, the atlas-image is resampled corresponding to this new affine.
                target_affine can be a 3x3 or a 4x4 matrix.

            target_shape:
                If specified, the atlas-image will be resized to match this new shape.
                len(target_shape) must be equal to 3.
                If target_shape is specified, a target_affine of shape (4, 4) must also be given.

            mask_threshold:
                Threshold for defining background.

        """
        if (mask_name, str(target_affine), str(target_shape)) not in AtlasLibrary.LIBRARY:
            self._add_mask_to_library(mask_name, target_affine, target_shape, mask_threshold)

        return AtlasLibrary.LIBRARY[(mask_name, str(target_affine), str(target_shape), str(mask_threshold))]

    def list_rois(self, atlas: str):
        """
        ROI listing of specific atlas.

        Parameters:
            atlas:
                Name of atlas.

        Returns:
            List of all ROIs for given Atlas.

        ToDo: Check if custom ATLAS should warn, too.

        """
        if atlas not in self.ATLAS_DICTIONARY.keys():
            msg = 'Atlas {} is not supported.'.format(atlas)
            logger.warning(msg)
            warnings.warn(msg)
            roi_names = []
        else:
            atlas = self.get_atlas(atlas)
            roi_names = [roi.label for roi in atlas.roi_list]
        return roi_names

    def _add_atlas_to_library(self,
                              atlas_name: str,
                              target_affine: Union[np.ndarray, None] = None,
                              target_shape: Union[tuple, list, None] = None,
                              mask_threshold: Union[float, None] = None,
                              interpolation: str = 'nearest') -> None:
        """
        Loading Atlas into the Library by name.

        Todo: find solution for multiprocessing spaming
        """
        # load atlas object from photon_atlasses
        if atlas_name in self.photon_atlases.keys():
            original_atlas_object = self.photon_atlases[atlas_name]
        else:
            original_atlas_object = self._check_custom_atlas(atlas_name)

        # now create new atlas object with different affine, shape and mask_threshold
        atlas_object = AtlasObject(name=original_atlas_object.name,
                                   path=original_atlas_object.path,
                                   labels_file=original_atlas_object.labels_file,
                                   mask_threshold=mask_threshold,
                                   affine=target_affine,
                                   shape=target_shape)

        # load atlas image
        img = image.load_img(atlas_object.path)

        # resample image by (target_affine, target_shape, interpolation)
        if target_affine is not None or target_shape is not None:
            img = image.resample_img(img,
                                     target_affine=target_affine,
                                     target_shape=target_shape,
                                     interpolation=interpolation)
            assert self._check_orientations(mask=img, target_affine=target_affine)
        atlas_object.atlas = img
        atlas_object.map = atlas_object.atlas.get_fdata()

        # apply mask threshold
        if mask_threshold is not None:
            atlas_object.map[atlas_object.map < mask_threshold] = 0
            atlas_object.map = atlas_object.map.astype(int)

        # now get indices
        atlas_object.indices = list(np.unique(atlas_object.map))

        # check labels
        if Path(atlas_object.labels_file).is_file():  # if we have a file with indices and labels
            labels = pd.read_table(atlas_object.labels_file, header=None)
            labels_dict = pd.Series(labels.iloc[:, 1].values, index=labels.iloc[:, 0]).to_dict()

            # check if background has been defined in labels.txt
            if 0 not in labels_dict.keys() and 0 in atlas_object.indices:
                # add 0 as background
                labels_dict[0] = 'Background'

            # check if map indices correspond with indices in the labels file
            if not sorted(atlas_object.indices) == sorted(list(labels_dict.keys())):
                logger.error("""
                The indices in map image ARE NOT the same as those in your *_labels.txt! Ignoring *_labels.txt.
                MapImage: 
                {}
                File: 
                {}
                """.format(str(sorted(atlas_object.indices)), str(sorted(list(labels_dict.keys())))))

                atlas_object.roi_list = [RoiObject(index=i, label=str(i), size=np.sum(i == atlas_object.map)) for i in
                                         atlas_object.indices]
            else:
                for i in range(len(atlas_object.indices)):
                    roi_index = atlas_object.indices[i]
                    new_roi = RoiObject(index=roi_index, label=labels_dict[roi_index].replace('\n', ''),
                                        size=np.sum(roi_index == atlas_object.map))
                    atlas_object.roi_list.append(new_roi)

        else:  # if we don't have a labels file, we just use str(indices) as labels
            atlas_object.roi_list = [RoiObject(index=i, label=str(i), size=np.sum(i == atlas_object.map)) for i in
                                     atlas_object.indices]

        # set new mask instance for non-empty ROI
        for roi in atlas_object.roi_list:
            self._add_mask_for_roi(atlas_object, roi)

        # finally add atlas to atlas library
        AtlasLibrary.LIBRARY[(atlas_name, str(target_affine), str(target_shape), str(mask_threshold))] = atlas_object
        logger.debug("BrainAtlas: Done adding atlas to library!")

    def _add_mask_to_library(self,
                             mask_name: str = '',
                             target_affine=None,
                             target_shape=None,
                             mask_threshold=0.5,
                             interpolation: str = 'nearest'):
        # Todo: find solution for multiprocessing spaming

        if mask_name in self.photon_masks.keys():
            original_mask_object = self.photon_masks[mask_name]
        else:
            logger.debug("Checking custom mask")
            original_mask_object = self._check_custom_mask(mask_name)

        mask_object = MaskObject(name=mask_name, mask_file=original_mask_object.mask_file)

        #mask_object.mask = image.threshold_img(mask_object.mask_file, threshold=mask_threshold)
        if mask_threshold is not None:
            mask_object.mask = image.math_img('img > {}'.format(mask_threshold), img=mask_object.mask_file)
        else:
            mask_object.mask = image.threshold_img(mask_object.mask_file, threshold=1.0)

        if target_affine is not None and target_shape is not None:
            mask_object.mask = image.resample_img(mask_object.mask,
                                                  target_affine=target_affine,
                                                  target_shape=target_shape,
                                                  interpolation=interpolation)
            assert self._check_orientations(mask=mask_object.mask, target_affine=target_affine)

        # check if roi is empty
        if np.sum(mask_object.mask.dataobj != 0) == 0:
            mask_object.is_empty = True
            msg = 'No voxels in mask after resampling ( {} ).'.format(mask_object.name)
            logger.error(msg)
            raise ValueError(msg)

        AtlasLibrary.LIBRARY[(mask_object.name,
                              str(target_affine),
                              str(target_shape),
                              str(mask_threshold))] = mask_object
        logger.debug("BrainMask: Done adding mask to library!")

    @staticmethod
    def _add_mask_for_roi(atlas_object: AtlasObject, roi) -> None:
        """
        check for empty ROIs and create roi mask
        """
        if roi.size == 0:
            return
        roi.mask = image.new_img_like(atlas_object.path, atlas_object.map == roi.index)

    def _load_photon_masks(self) -> dict:
        """
        Intern function for creating MaskObjects for every available Mask in the MASK_DICTIONARY.

        Returns:
            Mask by mask_id.

        """
        dir_atlases = path.join(path.dirname(inspect.getfile(AtlasLibrary)), 'atlases')
        photon_masks = dict()
        for mask_id, mask_info in self.MASK_DICTIONARY.items():
            mask_file = glob.glob(path.join(dir_atlases, path.join('*', mask_info)))[0]
            photon_masks[mask_id] = MaskObject(name=mask_id, mask_file=mask_file)
        return photon_masks

    def _load_photon_atlases(self) -> dict:
        """
        Intern function for creating AtlasObjects for every available Atlas in the ATLAS_DICTIONARY.

        Returns:
            Dict atlas by atlas_id.

        """
        dir_atlases = path.join(path.dirname(inspect.getfile(AtlasLibrary)), 'atlases')
        photon_atlases = dict()
        for atlas_id, atlas_info in self.ATLAS_DICTIONARY.items():
            atlas_file = glob.glob(path.join(dir_atlases, path.join('*', atlas_info)))[0]
            atlas_basename = path.basename(atlas_file)[:-7]
            atlas_dir = path.dirname(atlas_file)
            photon_atlases[atlas_id] = AtlasObject(name=atlas_id, path=atlas_file,
                                                   labels_file=path.join(atlas_dir, atlas_basename + '_labels.txt'))
        return photon_atlases

    @staticmethod
    def _check_orientations(mask, target_affine):
        orient_data = ''.join(nib.aff2axcodes(target_affine))
        orient_roi = ''.join(nib.aff2axcodes(mask.affine))
        if not orient_roi == orient_data:
            msg = 'Orientation of mask and data are not the same: ' \
                  '{0} (mask) vs. {1} (data)'.format(orient_roi, orient_data)
            logger.error(msg)
            raise ValueError(msg)
        return True

    @staticmethod
    def _check_custom_mask(mask_file: str):
        if not path.isfile(mask_file):
            msg = "Cannot find custom mask {}".format(mask_file)
            logger.error(msg)
            raise FileNotFoundError(msg)
        return MaskObject(name=mask_file, mask_file=mask_file)

    @staticmethod
    def _check_custom_atlas(atlas_file: str):
        logger.debug("Checking custom atlas")
        if not path.isfile(atlas_file):
            msg = "Cannot find custom atlas {}".format(atlas_file)
            logger.error(msg)
            raise FileNotFoundError(msg)
        labels_file = path.split(atlas_file)[0] + '_labels.txt'
        if not path.isfile(labels_file):
            msg = "Didn't find .txt file with ROI labels. Using indices as labels."
            logger.warning(msg)
            warnings.warn(msg)
        return AtlasObject(name=atlas_file, path=atlas_file, labels_file=labels_file)

    @staticmethod
    def find_rois_by_label(atlas_obj: AtlasObject, query_list: list) -> list:
        """
        Returns all ROIs of given AtlasObject with roi_label in query_list.

        Parameters:
            atlas_obj:
                AtlasObject, the object we are searching in

            query_list:
                Serach of ROI labels.

        Returns:
            List of ROI-lists for given label.

        """
        return [i for i in atlas_obj.roi_list if i.label in query_list]

    @staticmethod
    def find_rois_by_index(atlas_obj: AtlasObject, query_list: list) -> list:
        """
        Returns all ROIs of given AtlasObject with roi_index in query_list.

        Parameters:
            atlas_obj:
                AtlasObject, the object we are searching in.

            query_list:
                Search list of indices.

        Returns:
            List of ROI-lists for given index.
        """
        return [i for i in atlas_obj.roi_list if i.index in query_list]

    @staticmethod
    def _get_nii_files_from_folder(folder_path: str, extension: str=".nii.gz"):
        """Returns all file with given extension in folder path."""
        return glob.glob(folder_path + '*' + extension)
