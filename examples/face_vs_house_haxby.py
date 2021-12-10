import os
import warnings
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_haxby
from sklearn.model_selection import ShuffleSplit, KFold
from nilearn.image import index_img

from photonai.base import Hyperpipe, PipelineElement, Preprocessing
from photonai_neuro import NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)


haxby_dataset = fetch_haxby()
func_img = haxby_dataset.func[0]

# Load target information as string and give a numerical identifier to each
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral['labels']

# Restrict the analysis to faces and places
condition_mask = behavioral['labels'].isin(['face', 'house'])
conditions = conditions[condition_mask]
func_img = index_img(func_img, condition_mask)

# Confirm that we now have 2 conditions
print(conditions.unique())

X = np.array([func_img.slicer[:, :, :, i] for i in range(func_img.shape[3])])
y = conditions.values

base_folder = os.path.dirname(os.path.abspath(__file__))
cache_folder_path = os.path.join(base_folder, "cache2")
tmp_folder_path = os.path.join(base_folder, "tmp")

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('HaxbyClassifier',
                 optimizer='grid_search',
                 metrics=['balanced_accuracy', 'accuracy', 'f1_score'],
                 best_config_metric='balanced_accuracy',
                 inner_cv=KFold(n_splits=3),
                 outer_cv=ShuffleSplit(n_splits=2),
                 project_folder=tmp_folder_path,
                 verbosity=1,
                 #cache_folder=cache_folder_path,
                 use_test_set=True)

# ToDo important cache_folder bug
# an activate cache_folder reason in some strange problems

preproc = Preprocessing()
preproc += PipelineElement('LabelEncoder')
pipe += preproc

neuro_branch = NeuroBranch("NB", nr_of_processes=1)
neuro_branch += PipelineElement('BrainMask', mask_image=haxby_dataset.mask, extract_mode='vec')
pipe += neuro_branch

pipe += PipelineElement("StandardScaler")
pipe += PipelineElement('SelectPercentile', percentile=5)
pipe += PipelineElement('LinearSVC', penalty='l2', max_iter=1e4)

pipe.fit(X[:150], y[:150])  # <- split up some data for investigate the cache_folder bug (optimum pipe not affected).

_, real, _ = pipe.preprocessing.elements[0].transform(X, conditions[150:])

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
print(confusion_matrix(y_true=real, y_pred=pipe.predict(func_img.slicer[:,:,:,150:])))
print(balanced_accuracy_score(y_true=real, y_pred=pipe.predict(func_img.slicer[:,:,:,150:])))
