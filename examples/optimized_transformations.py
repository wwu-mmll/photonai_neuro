import os
import warnings
import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

from photonai.base import Hyperpipe, PipelineElement
from photonai_neuro import NeuroBranch
from photonai.optimization import IntegerRange, FloatRange

warnings.filterwarnings("ignore", category=DeprecationWarning)

# GET DATA FROM OASIS
n_subjects = 300
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)

base_folder = os.path.dirname(os.path.abspath(__file__))
cache_folder_path = os.path.join(base_folder, "cache")
tmp_folder_path = os.path.join(base_folder, "tmp")

nr_of_outer = 1
# DESIGN YOUR PIPELINE
pipe = Hyperpipe('OptimizedTransform',
                 optimizer='sk_opt',
                 optimizer_params={'n_initial_points': 5, 'initial_point_generator': 'sobol', 'n_configurations': 20},
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 #outer_cv=ShuffleSplit(n_splits=nr_of_outer, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=4, test_size=0.2),
                 project_folder=tmp_folder_path,
                 verbosity=1,
                 #cache_folder=cache_folder_path,
                 use_test_set=True)

neuro_branch = NeuroBranch('NeuroBranch', nr_of_processes=2)
neuro_branch += PipelineElement('SmoothImages', hyperparameters={'fwhm': IntegerRange(2, 5)}, batch_size=20)
neuro_branch += PipelineElement('ResampleImages', hyperparameters={'voxel_size': FloatRange(2, 7)}, batch_size=20)
neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec', batch_size=20)
pipe += neuro_branch

pipe += PipelineElement("StandardScaler")
pipe += PipelineElement('SelectPercentile', percentile=15)
pipe += PipelineElement('LinearSVR', max_iter=1e4)

pipe.fit(X, y)

# visual comparison
results = []
for nr_outer in range(nr_of_outer):
    results.extend([(int(a.human_readable_config['NeuroBranch'][0].split("=")[1]),
                     float(a.human_readable_config['NeuroBranch'][1].split("=")[1]),
                     a.metrics_test[0].value) for a in pipe.results.outer_folds[0].tested_config_list])

ncols = len(list(set([x[0] for x in results])))
fig, ax = plt.subplots(nrows=1, ncols=ncols)

for i, col in enumerate(list(set([x[0] for x in results]))):
    ax[i].scatter(*zip(*[x[1:] for x in results if x[0] == col]))
    ax[i].title.set_text("fwhm=={}".format(col))
    ax[i].set_ylim(np.min([x[2] for x in results])-0.2, np.max([x[2] for x in results])+0.2)
plt.ylabel("MAE")
plt.show()
