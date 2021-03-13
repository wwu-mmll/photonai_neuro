import warnings
import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, Preprocessing
from photonai.optimization import FloatRange
from photonai_neuro import NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)

# GET DATA FROM OASIS
n_subjects = 300
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)

pipe = Hyperpipe('GrayMatter',
                 optimizer='sk_opt',
                 optimizer_params={'n_initial_points': 8, 'initial_point_generator': 'sobol', 'n_configurations': 15},
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=2),
                 inner_cv=ShuffleSplit(n_splits=3),
                 verbosity=1,
                 cache_folder="./cache",
                 use_test_set=True,
                 project_folder='./tmp/')

# CHOOSE BETWEEN AVAILABLE MASKS:
# 'MNI_ICBM152_GrayMatter'
# 'MNI_ICBM152_WhiteMatter'
# 'MNI_ICBM152_WholeBrain'
# 'Cerebellum'

preproc = Preprocessing()
neuro_branch = NeuroBranch("NB", nr_of_processes=2)
mask = PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec')
neuro_branch += mask
preproc += neuro_branch
pipe += preproc

pipe += PipelineElement("StandardScaler")
pipe += PipelineElement('SelectPercentile', hyperparameters={'percentile': FloatRange(5, 12)})
pipe += PipelineElement('LinearSVR', hyperparameters={'C': FloatRange(0.1, 50)}, max_iter=1e4)
pipe.fit(X, y)
