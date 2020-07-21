import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Branch
from photonai_neuro import AtlasStacker, NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Specify where the results should be written to and the name of your analysis
results_folder = './tmp/'
cache_folder = './tmp/cache'

# GET DATA FROM OASIS
n_subjects = 20
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
gender = dataset_files.ext_vars['mf'].astype(str)
y = np.array(gender)
X = np.array(dataset_files.gray_matter_maps)


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
settings = OutputSettings(project_folder=results_folder)


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('atlas_stacker_example',
                    optimizer='grid_search',
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    inner_cv=KFold(n_splits=2),
                    verbosity=2,
                    output_settings=settings,
                    cache_folder=cache_folder)

roi_branch = Branch("ROI")
roi_branch += PipelineElement("PCA", n_components=0.5)
final_estimator = PipelineElement('SVC')


# DEFINE NEURO ELEMENTS
#brain_atlas = PipelineElement('BrainAtlas', atlas_name="Yeo_7", extract_mode='vec',
#                                rois='all', batch_size=200)
brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL",
                              rois=['Hippocampus_L', 'Hippocampus_R', "Frontal_Sup_Orb_L"], batch_size=200)

neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += brain_atlas

# NOW TRAIN ATLAS MAPPER
atlas_stacker = AtlasStacker(neuro_element=neuro_branch,
                            hyperpipe=my_pipe,
                            folder=results_folder,
                            roi_branch=roi_branch,
                            final_estimator=final_estimator)
atlas_stacker.fit(X, y)


#-- alternativ atlas stacker
"""
import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Preprocessing, Branch, Stack
from photonai_neuro import NeuroBranch, BrainMask

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Specify where the results should be written to and the name of your analysis
results_folder = './tmp/'
cache_folder = './tmp/cache'

# GET DATA FROM OASIS
n_subjects = 50
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
gender = dataset_files.ext_vars['mf'].astype(str)
y = np.array(gender)
X = np.array(dataset_files.gray_matter_maps)


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
settings = OutputSettings(project_folder=results_folder)


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('atlas_mapper_example',
                    optimizer='grid_search',
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    inner_cv=KFold(n_splits=2),
                    verbosity=2,
                    output_settings=settings,
                    cache_folder=cache_folder)


# DEFINE NEURO ELEMENTS
affine, shape = BrainMask.get_format_info_from_first_image(X[0])
brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL",
                              rois=['Hippocampus_L', 'Hippocampus_R', "Frontal_Sup_Orb_L"],
                              example_data=[X[0]],
                              batch_size=200)
neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += brain_atlas

preprocessing = Preprocessing()
preprocessing += brain_atlas
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing

stack = Stack("ROI_Stack")
for roi in ['Hippocampus_L', 'Hippocampus_R']:
    branch = Branch(roi+"_branch")
    branch += PipelineElement("RoiFilter",
                              roi_allocation=brain_atlas.base_element.roi_allocation,
                              rois=roi,
                              mask_indices=brain_atlas.base_element.mask_indices)
    branch += PipelineElement('SVC')
    stack += branch
my_pipe += stack

my_pipe += PipelineElement("PhotonVotingClassifier")

my_pipe.fit(X, y)
"""