import os
import warnings
import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai_neuro import NeuroBranch
from photonai.optimization import IntegerRange, FloatRange

warnings.filterwarnings("ignore", category=DeprecationWarning)

# GET DATA FROM OASIS
n_subjects = 150
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)

base_folder = os.path.dirname(os.path.abspath(__file__))
cache_folder_path = os.path.join(base_folder, "cache")
tmp_folder_path = os.path.join(base_folder, "tmp")

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Limbic_System',
                 optimizer='sk_opt',
                 optimizer_params={'n_initial_points': 5, 'initial_point_generator': 'sobol'},
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=2, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                 project_folder=tmp_folder_path,
                 verbosity=1,
                 cache_folder=cache_folder_path,
                 use_test_set=True)

batch_size = 25
neuro_branch = NeuroBranch('NeuroBranch', nr_of_processes=4)
neuro_branch += PipelineElement('SmoothImages', hyperparameters={'fwhm': IntegerRange(2, 5)},
                                batch_size=batch_size)
neuro_branch += PipelineElement('ResampleImages', hyperparameters={'voxel_size': FloatRange(3, 6)},
                                batch_size=batch_size)
neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec',
                                batch_size=batch_size)

pipe += neuro_branch

#pipe += PipelineElement('PCA', n_components=40)
pipe += PipelineElement('LinearSVR', hyperparameters={'C': FloatRange(0.5, 2)})

pipe.fit(X, y)
