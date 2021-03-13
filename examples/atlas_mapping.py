import warnings
import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, Preprocessing
from photonai_neuro import AtlasMapper, NeuroBranch

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

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('atlas_mapper_example',
                    optimizer='grid_search',
                    metrics=['accuracy', 'balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    inner_cv=ShuffleSplit(n_splits=10),
                    verbosity=1,
                    project_folder=results_folder,
                    cache_folder=cache_folder)

preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing
my_pipe += PipelineElement("StandardScaler")
my_pipe += PipelineElement('LinearSVC')


# DEFINE NEURO ELEMENTS
# brain_atlas = PipelineElement('BrainAtlas', atlas_name="Yeo_7", extract_mode='vec',
#                                rois='all', batch_size=200)
brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL",
                              rois=['Hippocampus_L', 'Hippocampus_R', "Frontal_Sup_Orb_L"], batch_size=200)

neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += brain_atlas

# NOW TRAIN ATLAS MAPPER
atlas_mapper = AtlasMapper(neuro_branch, my_pipe, results_folder, create_surface_plots=True)
# atlas_mapper.generate_mappings()
atlas_mapper.fit(X, y)

# LOAD TRAINED ATLAS MAPPER AND PREDICT
atlas_mapper = AtlasMapper.load_from_folder(folder=results_folder, analysis_name='atlas_mapper_example')
# you can either load an atlas mapper by specifying the atlas_mapper_meta.json file that has been created during fit()
# or simply specify the results folder in which your model was saved (and you can also specify the analysis name in case
# there are multiple atlas mapper within one folder)

# atlas_mapper.load_from_file(os.path.join(results_folder) + 'atlas_mapper_meta.json')
# atlas_mapper.load_from_folder(folder=results_folder, analysis_name='atlas_mapper_example')
print(atlas_mapper.predict(X))
