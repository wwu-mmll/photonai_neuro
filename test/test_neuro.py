import os
import numpy as np
from nilearn import image
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai.processing import ResultsHandler
from photonai.helper.photon_base_test import PhotonBaseTest

from photonai_neuro import NeuroBranch


class NeuroBaseTest(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(NeuroBaseTest, cls).setUpClass()

    @staticmethod
    def get_data_from_oasis(n_subjects=10):
        # GET DATA FROM OASIS
        dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
        age = dataset_files.ext_vars['age'].astype(float)
        return np.array(dataset_files.gray_matter_maps), np.array(age)

    def setUp(self):
        super(NeuroBaseTest, self).setUp()
        self.test_folder = os.path.dirname(os.path.abspath(__file__))
        self.atlas_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../photonai_neuro/atlases/')
        self.atlas_name = "AAL"
        self.roi_list = ["Hippocampus_R", "Hippocampus_L", "Amygdala_L", "Amygdala_R"]
        self.X, self.y = self.get_data_from_oasis()


class NeuroTests(NeuroBaseTest):

    def test_inverse_transform(self):
        pipe = Hyperpipe('Limbic_System',
                         optimizer='grid_search',
                         metrics=['mean_absolute_error'],
                         best_config_metric='mean_absolute_error',
                         inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                         verbosity=0,
                         cache_folder=self.cache_folder_path,
                         use_test_set=True,
                         project_folder=self.tmp_folder_path)

        atlas = PipelineElement('BrainAtlas',
                                rois=['Hippocampus_L', 'Amygdala_L'],
                                atlas_name="AAL", extract_mode='vec', batch_size=20)

        neuro_branch = NeuroBranch('NeuroBranch')
        neuro_branch += atlas
        pipe += neuro_branch

        pipe += PipelineElement('LinearSVR')

        pipe.fit(self.X, self.y)

        handler = ResultsHandler(pipe.results)
        importance_scores_optimum_pipe = handler.results.best_config_feature_importances

        manual_img, _, _ = pipe.optimum_pipe.inverse_transform(importance_scores_optimum_pipe, None)
        img = image.load_img(os.path.join(pipe.results_handler.output_settings.results_folder,
                                          'optimum_pipe_feature_importances_backmapped.nii.gz'))
        self.assertTrue(np.array_equal(manual_img.get_data(), img.get_data()))
