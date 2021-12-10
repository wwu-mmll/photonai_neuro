import warnings
import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

from photonai.base import Hyperpipe, PipelineElement, Switch, Preprocessing
from photonai_neuro import NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)


"""
In this example, we show the necessity of scaling the data. 
We search for the best scaler with PHOTONAIs Switch element.
Easy determination by trial and error via Grid Search. 
"""


class NoScaling(BaseEstimator, TransformerMixin):

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        return X

# GET DATA FROM OASIS
n_subjects = 400
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)

nr_of_outer = 3

pipe = Hyperpipe('BestScaling',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error', 'mean_squared_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=nr_of_outer, test_size=0.2),
                 inner_cv=KFold(n_splits=3),
                 verbosity=1,
                 cache_folder="./cache",
                 use_test_set=True,
                 project_folder='./tmp/')

# ToDo activate cache_folder in Preprocessing element.
preproc = Preprocessing()
neuro_branch = NeuroBranch("NB", nr_of_processes=2)
neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec')
preproc += neuro_branch
pipe += preproc

scaling_switch = Switch("scaling_switch")
scaling_switch += PipelineElement("MaxAbsScaler")
scaling_switch += PipelineElement("MinMaxScaler")
scaling_switch += PipelineElement("StandardScaler")
scaling_switch += PipelineElement("RobustScaler")
scaling_switch += PipelineElement.create('NoScaling', base_element=NoScaling(), hyperparameters={})
pipe += scaling_switch

pipe += PipelineElement('SelectPercentile', percentile=25)
pipe += PipelineElement('LinearSVR', max_iter=2e4)
pipe.fit(X, y)

# visual comparison
results = []
for nr_outer in range(nr_of_outer):
    results.extend([(a.human_readable_config['scaling_switch'][0], a.metrics_test[0].value)
                    for a in pipe.results.outer_folds[nr_outer].tested_config_list])
fig = plt.figure()
plt.scatter(*zip(*results))
plt.xlabel('Scaler', fontsize=18)
plt.ylabel('MAE', fontsize=18)
plt.show()
