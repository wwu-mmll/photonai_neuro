import os
import numpy as np
from nilearn import image
import random
import warnings

from photonai.base import PipelineElement
from photonai.photonlogger import logger

from photonai_neuro import AtlasLibrary, BrainAtlas
from photonai_neuro.objects import NiftiConverter
from test.test_neuro import NeuroBaseTest


class BrainAtlasTests(NeuroBaseTest):

    def test_brain_atlas_mean(self):

        brain_atlas = BrainAtlas(self.atlas_name, "vec", rois=self.roi_list)
        vec_output = brain_atlas.transform(self.X)
        self.assertEqual(len(self.X), vec_output.shape[0])
        self.assertEqual(vec_output.shape[1], brain_atlas.mask_indices.shape[0])

        brain_atlas_mean = BrainAtlas(self.atlas_name, "mean", rois='all')
        mean_rois = brain_atlas_mean.transform(self.X)
        self.assertEqual(mean_rois.shape[0], len(self.X))
        # todo: should be 109 because of mean, but is the same as vec.
        # todo: why? all ROIs == 116, 117 with background, but is not calculated in transform (_get_rois removed it.)
        self.assertEqual(mean_rois.shape[1], len(dict(AtlasLibrary.LIBRARY)[(self.atlas_name,
                                                                             str(brain_atlas_mean.affine),
                                                                             str(brain_atlas_mean.shape),
                                                                             str(brain_atlas_mean.mask_threshold))]
                                                 .roi_list) - 1)

    def test_brain_atlas_load(self):

        brain_atlas = AtlasLibrary().get_atlas(self.atlas_name)

        # manually load brain atlas
        man_map = image.load_img(os.path.join(self.atlas_folder, 'AAL_SPM12/AAL.nii.gz')).get_data()
        self.assertTrue(np.array_equal(man_map, brain_atlas.map))

    def test_custom_atlas(self):
        custom_atlas = os.path.join(self.atlas_folder, 'AAL_SPM12/AAL.nii.gz')

        # todo: consider renaming atlas_name to only atlas as we are giving a full path here which might be confusing?
        atlas = PipelineElement('BrainAtlas', atlas_name=custom_atlas, extract_mode='vec', batch_size=20)
        _ = atlas.transform(self.X)

        with self.assertRaises(FileNotFoundError):
            atlas = PipelineElement('BrainAtlas', atlas_name='XXXXX', extract_mode='vec', batch_size=20)
            atlas.transform(self.X)

    def test_atlases(self):
        for atlas in AtlasLibrary().ATLAS_DICTIONARY.keys():
            self.run_atlas(atlas)
            self.validity_check_roi_extraction(atlas)
            AtlasLibrary.LIBRARY = {}

    def run_atlas(self, atlas):
        print("Running tests for atlas {}".format(atlas))
        brain_atlas = PipelineElement('BrainAtlas', atlas_name=atlas, extract_mode='mean')
        brain_atlas.transform(self.X[:2])

    def validity_check_roi_extraction(self, atlas):
        affine, shape = NiftiConverter.get_format_info_from_first_image(self.X)
        logger.debug("Checking atlas {}".format(atlas))
        rois = AtlasLibrary().get_atlas(atlas, affine, shape).roi_list[1:3]
        rois = [roi.label for roi in rois]
        brain_atlas = BrainAtlas(atlas_name=atlas)
        brain_atlas.rois = rois
        X_t = brain_atlas.transform(self.X[:2])

        "-".join(rois)
        name = os.path.join(self.test_folder, atlas + '_' + "-".join(rois))
        # todo: what good does that do?
        brain_atlas._validity_check_roi_extraction(X_t[0], filename=name)
        self.assertTrue(os.path.exists(name+".nii"))
        os.remove(name+".nii")

    def test_roi_indices(self):
        for _ in range(10):
            roi_list_rand_order = ["Hippocampus_L", "Hippocampus_R", "Amygdala_L", "Amygdala_R"]
            random.shuffle(roi_list_rand_order)
            atlas = BrainAtlas(atlas_name=self.atlas_name,
                               extract_mode='vec',
                               rois=roi_list_rand_order)
            atlas.transform(self.X[:2])
            self.assertListEqual(list(atlas.roi_allocation.keys()),
                                 ["Hippocampus_L", "Hippocampus_R", "Amygdala_L", "Amygdala_R"])

    def test_different_inputs(self):
        custom_atlas = os.path.join(self.atlas_folder, 'AAL_SPM12/AAL.nii.gz')

        for x in [self.X[0], self.X[:2], image.load_img(self.X[0])]:
            atlas = PipelineElement('BrainAtlas', atlas_name=custom_atlas, extract_mode='vec', batch_size=20)
            self.assertIsInstance(atlas.transform(x)[0], np.ndarray)

        with self.assertRaises(ValueError):
            atlas = PipelineElement('BrainAtlas', atlas_name=custom_atlas, extract_mode='vec', batch_size=20)
            atlas.transform(X=5)

    def test_false_collection_mode(self):
        custom_atlas = os.path.join(self.atlas_folder, 'AAL_SPM12/AAL.nii.gz')
        with self.assertRaises(ValueError):
            atlas = PipelineElement('BrainAtlas', atlas_name=custom_atlas, extract_mode='vec', batch_size=20)
            atlas.base_element.collection_mode = "array"
            atlas.transform(self.X)

    def test_list_rois(self):
        self.assertTrue(AtlasLibrary().list_rois("Schaefer2018_100Parcels_7Networks"))
        with warnings.catch_warnings(record=True) as w:
            AtlasLibrary().list_rois("plAtlas")
            assert len(w) == 1

    def test_atlas_threshold(self):
        roi_list = ["Hippocampus_L", "Hippocampus_R", "Amygdala_L", "Amygdala_R"]
        atlas_wo_threshold = BrainAtlas(atlas_name=self.atlas_name,
                                        extract_mode='vec',
                                        rois=roi_list)
        atlas_w_threshold = BrainAtlas(atlas_name=self.atlas_name,
                                       extract_mode='vec',
                                       rois=roi_list,
                                       mask_threshold=3000)
        a = atlas_wo_threshold.transform(self.X[:2])
        b = atlas_w_threshold.transform(self.X[:2])

        self.assertEqual(a.shape[0], b.shape[0])
        self.assertGreater(a.shape[1], b.shape[1])
