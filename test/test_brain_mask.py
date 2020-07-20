import os
import numpy as np
from nilearn.input_data import NiftiMasker

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
        mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='vec', batch_size=20)
        _ = mask.transform(self.X)

        mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='mean', batch_size=20)
        _ = mask.transform(self.X)

        mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='box', batch_size=20)
        _ = mask.transform(self.X)

        with self.assertRaises(FileNotFoundError):
            mask = PipelineElement('BrainMask', mask_image='XXXXX', extract_mode='vec', batch_size=20)
            mask.transform(self.X)

    def test_all_masks(self):
        for mask in AtlasLibrary().MASK_DICTIONARY.keys():
            brain_mask = PipelineElement('BrainMask', mask_image=mask, extract_mode='vec')
            brain_mask.transform(self.X)