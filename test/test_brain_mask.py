import os
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
from nibabel.nifti1 import Nifti1Image

from photonai.base import PipelineElement

from photonai_neuro import BrainMask, AtlasLibrary, BrainAtlas
from test.test_neuro import NeuroBaseTest


class BrainMaskTests(NeuroBaseTest):

    def test_brain_masker(self):

        affine, shape = BrainMask.get_format_info_from_first_image(self.X)
        atlas_obj = AtlasLibrary().get_atlas(self.atlas_name, affine, shape)
        roi_objects = BrainAtlas._get_rois(atlas_obj, which_rois=self.roi_list, background_id=0)

        for roi in roi_objects:
            masker = BrainMask(mask_image=roi, affine=affine, shape=shape, extract_mode="vec")
            own_calculation = masker.transform(self.X[0])
            nilearn_func = NiftiMasker(mask_img=roi.mask, target_affine=affine, target_shape=shape, dtype='float32')
            nilearn_calculation = nilearn_func.fit_transform(self.X[0])

            self.assertTrue(np.array_equal(own_calculation, nilearn_calculation))

    def test_custom_mask(self):
        custom_mask = os.path.join(self.atlas_folder, 'Cerebellum/P_08_Cere.nii.gz')
        for em in ['vec', 'mean', 'box', 'img']:
            mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode=em, batch_size=20)
            result = mask.transform(self.X)
            self.assertIsInstance(result, tuple)
            if not em == 'img':
                self.assertIsInstance(result[0], np.ndarray)
            # todo: check vec, mean and box for shapes
            else:
                self.assertIsInstance(result[0], Nifti1Image)

        with self.assertRaises(NameError):
            mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='circle', batch_size=20)
            mask.transform(self.X)

        with self.assertRaises(FileNotFoundError):
            mask = PipelineElement('BrainMask', mask_image='XXXXX', extract_mode='vec', batch_size=20)
            mask.transform(self.X)

    def test_all_masks(self):
        for mask in AtlasLibrary().MASK_DICTIONARY.keys():
            brain_mask = PipelineElement('BrainMask', mask_image=mask, extract_mode='vec')
            brain_mask.transform(self.X)

    def test_get_info(self):
        for x in [self.X, self.X[0], image.load_img(self.X[0])]:
            affine, shape = BrainMask.get_format_info_from_first_image(x)
            self.assertIsInstance(affine, np.ndarray)
            self.assertIsInstance(shape, tuple)

        with self.assertRaises(ValueError):
            BrainMask.get_format_info_from_first_image(42)

    def test_inverse(self):
        custom_mask = os.path.join(self.atlas_folder, 'Cerebellum/P_08_Cere.nii.gz')
        for em in ['mean', 'box', 'img']:
            with self.assertRaises(NotImplementedError):
                mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode=em, batch_size=20)
                result = mask.transform(self.X)
                mask.inverse_transform(result)

        mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='vec', batch_size=20)
        result = mask.transform(self.X[0:1])
        back_transformed = mask.inverse_transform(result[0])[0]
        self.assertIsInstance(back_transformed, Nifti1Image)
        # todo: check shape: that it is equal to mask?
