import os
import pandas as pd
from sklearn.model_selection import KFold
from nilearn.datasets import fetch_haxby
from nilearn.image import index_img, iter_img

from photonai.base import Hyperpipe, PipelineElement

data_files = fetch_haxby()

# Load behavioral data
behavioral = pd.read_csv(data_files.session_target[0], sep=" ")

conditions = behavioral['labels']
condition_mask = conditions.isin(['face', 'house'])

func_filenames = data_files.func[0]
X = index_img(func_filenames, condition_mask)
X = [x for x in iter_img(X)]
y = conditions[condition_mask]
y = (y == 'face').astype(int)

# DEFINE OUTPUT SETTINGS
base_folder = os.path.dirname(os.path.abspath(__file__))
cache_folder_path = os.path.join(base_folder, "cache")
tmp_folder_path = os.path.join(base_folder, "tmp")

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Face_House',
                 optimizer='grid_search',
                 metrics=['accuracy', 'f1_score'],
                 best_config_metric='f1_score',
                 inner_cv=KFold(n_splits=2, shuffle=True),
                 project_folder=tmp_folder_path,
                 verbosity=1,
                 cache_folder=cache_folder_path,
                 use_test_set=True)

pipe += PipelineElement('SpaceNetClassifier',
                        hyperparameters={'penalty': ['graph-net', 'tv-l1']},
                        screening_percentile=5.,
                        verbose=0
                        )

pipe.fit(X, y)
