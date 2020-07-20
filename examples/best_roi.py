import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Preprocessing, Branch, Stack
from photonai.optimization import Categorical
from photonai_neuro import NeuroBranch, BrainMask

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Specify where the results should be written to and the name of your analysis
results_folder = './tmp/'
cache_folder = './tmp/cache'

# GET DATA FROM OASIS
n_subjects = 400
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
gender = dataset_files.ext_vars['mf'].astype(str)
y = np.array(gender)
X = np.array(dataset_files.gray_matter_maps)


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
settings = OutputSettings(project_folder=results_folder)


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('Best_ROI',
                    optimizer='grid_search',
                    metrics=['accuracy', 'f1_score'],
                    best_config_metric='f1_score',
                    inner_cv=KFold(n_splits=3),
                    outer_cv=KFold(n_splits=3),
                    verbosity=2,
                    output_settings=settings,
                    cache_folder=cache_folder)


# DEFINE NEURO ELEMENTS
affine, shape = BrainMask.get_format_info_from_first_image(X[0])
#brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL",
#                              rois=['Hippocampus_L', 'Hippocampus_R', "Frontal_Sup_Orb_L"],
#                              example_data=[X[0]],
#                              batch_size=200)
available_rois = ['Network_1', 'Network_2', 'Network_3', 'Network_4', 'Network_5', 'Network_6', 'Network_7']
brain_atlas = PipelineElement('BrainAtlas',
                              atlas_name="Yeo_7",
                              extract_mode='vec',
                              example_data=[X[0]],
                              rois='all',
                              batch_size=200)
neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += brain_atlas

preprocessing = Preprocessing()
preprocessing += brain_atlas
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing

rois_cat = Categorical(available_rois+[available_rois[:4], available_rois[4:]]+[available_rois])

my_pipe += PipelineElement("RoiFilter",
                           hyperparameters={'rois': rois_cat},
                           roi_allocation=brain_atlas.base_element.roi_allocation,
                           mask_indices=brain_atlas.base_element.mask_indices)

my_pipe += PipelineElement('PCA', n_components=0.6)
my_pipe += PipelineElement('RandomForestClassifier')



my_pipe.fit(X, y)
print("test")
