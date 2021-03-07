import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai_neuro import NeuroBranch

# GET DATA FROM OASIS
n_subjects = 150
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# DESIGN YOUR PIPELINE
pipe = Hyperpipe('GrayMatter',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=2, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=2, test_size=0.2),
                 verbosity=0,
                 cache_folder="./cache",
                 use_test_set=False,
                 project_folder='./tmp/')

# CHOOSE BETWEEN MASKS
mask = PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec', batch_size=20)

# EITHER ADD A NEURO BRANCH OR THE ATLAS ITSELF
neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += mask
pipe += neuro_branch

pipe += PipelineElement('LinearSVR')

# since we're predicting age and age cannot be below 0 and some upper threshold like 90, we can restrict the SVR's
# range of predictions
pipe += PipelineElement('RangeRestrictor', low=16, high=90)

pipe.fit(X, y)
