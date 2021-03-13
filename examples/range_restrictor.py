import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

from photonai.base import Hyperpipe, PipelineElement
from photonai_neuro import NeuroBranch

# GET DATA FROM OASIS
n_subjects = 150
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)

nr_of_outer = 2
# DESIGN YOUR PIPELINE
pipe = Hyperpipe('GrayMatter',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=nr_of_outer, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=2, test_size=0.2),
                 verbosity=1,
                 cache_folder="./cache",
                 use_test_set=True,
                 project_folder='./tmp/')

neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec', batch_size=20)
pipe += neuro_branch

pipe += PipelineElement("StandardScaler")
pipe += PipelineElement('SelectPercentile', percentile=25)
pipe += PipelineElement('LinearSVR')

# Since we're predicting age, and age cannot be below 0
# and some upper threshold like 90, we can restrict the SVR's
# range of predictions.
pipe += PipelineElement('RangeRestrictor', low=18, high=90, test_disabled=True)

pipe.fit(X, y)

# visual comparison
results = []
for nr_outer in range(nr_of_outer):
    results.extend([(a.human_readable_config['RangeRestrictor'][0], a.metrics_test[0].value)
                    for a in pipe.results.outer_folds[nr_outer].tested_config_list])
fig = plt.figure()
plt.scatter(*zip(*results))
fig.subtitle('RangeRestrictor', fontsize=18)
plt.ylabel('MAE', fontsize=18)
plt.show()