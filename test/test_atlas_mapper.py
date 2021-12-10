import os
import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Preprocessing
from photonai_neuro import AtlasMapper, NeuroBranch
from test.test_neuro import NeuroBaseTest


class AtlasMapperTests(NeuroBaseTest):

    def create_hyperpipe(self):
        results_folder = self.tmp_folder_path
        cache_folder = self.cache_folder_path

        my_pipe = Hyperpipe('atlas_mapper_example',
                            optimizer='grid_search',
                            metrics=['accuracy'],
                            best_config_metric='accuracy',
                            inner_cv=KFold(n_splits=2),
                            verbosity=0,
                            project_folder=results_folder,
                            cache_folder=cache_folder)

        preprocessing = Preprocessing()
        preprocessing += PipelineElement("LabelEncoder")
        my_pipe += preprocessing
        my_pipe += PipelineElement('LinearSVC')
        return my_pipe

    @staticmethod
    def create_data():
        n_subjects = 20
        dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
        _ = dataset_files.ext_vars['age'].astype(float)
        gender = dataset_files.ext_vars['mf'].astype(str)
        y = np.array(gender)
        X = np.array(dataset_files.gray_matter_maps)
        return X, y

    def test_fit_reload_all(self):
        results_folder = './tmp/'
        X, y = self.create_data()
        brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL",
                                      rois=['Hippocampus_L', 'Hippocampus_R', "Frontal_Sup_Orb_L"], batch_size=200)

        my_pipe = self.create_hyperpipe()

        neuro_branch = NeuroBranch('NeuroBranch')
        neuro_branch += brain_atlas

        atlas_mapper = AtlasMapper(neuro_element=neuro_branch,
                                   hyperpipe=my_pipe,
                                   folder=results_folder,
                                   create_surface_plots=True)

        with self.assertRaises(Exception):
            atlas_mapper.predict(X)

        atlas_mapper.fit(X, y)
        result_apri = atlas_mapper.predict(X)

        # load from folder
        atlas_mapper = AtlasMapper.load_from_folder(folder=results_folder, analysis_name='atlas_mapper_example')
        result_apo0 = atlas_mapper.predict(X)

        # load from file
        atlas_mapper = AtlasMapper.load_from_file(file=results_folder+"atlas_mapper_example_atlas_mapper_meta.json")
        result_apo1 = atlas_mapper.predict(X)

        with self.assertRaises(ValueError):
            atlas_mapper.fit(X, y)

        for key in result_apri.keys():
            np.testing.assert_array_equal(result_apo0[key], result_apri[key])
            np.testing.assert_array_equal(result_apo1[key], result_apri[key])

        self.assertTrue(os.path.exists(results_folder+"importance_scores_surface.png"))

    def test_fit_concat(self):
        results_folder = self.tmp_folder_path
        X, y = self.create_data()
        brain_atlas = PipelineElement('BrainAtlas', atlas_name="Yeo_7", extract_mode='vec',
                                      rois='all', batch_size=200)
        my_pipe = self.create_hyperpipe()
        neuro_branch = NeuroBranch('NeuroBranch')
        neuro_branch += brain_atlas
        atlas_mapper = AtlasMapper(neuro_element=neuro_branch,
                                   hyperpipe=my_pipe,
                                   folder=results_folder,
                                   create_surface_plots=False)
        with self.assertRaises(Exception):
            atlas_mapper.predict(X)
        atlas_mapper.fit(X, y)
