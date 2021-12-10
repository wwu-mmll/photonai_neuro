import json
import os
from glob import glob
from typing import Union

import joblib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from nilearn import datasets, surface, plotting

from photonai.base import PipelineElement, Hyperpipe
from photonai.photonlogger.logger import logger
from photonai.processing import ResultsHandler

from photonai_neuro.brain_atlas import BrainAtlas, AtlasLibrary
from photonai_neuro.neuro_branch import NeuroBranch


class AtlasMapper:
    """Mapping between neuro processing and Hyperpipes for given regions of interest."""

    def __init__(self, neuro_element: Union[NeuroBranch, PipelineElement], hyperpipe: Hyperpipe = None,
                 folder: str = "./tmp/", create_surface_plots: bool = False):
        """
        Initialize the object.

        Parameters:
            neuro_element:
                Neuro processing in front of hyperpipe.

            hyperpipe:
                Hyperpipe to fit.

            folder:
                Output path for created hyperpipes and other results.

            create_surface_plots:
                Enable/Disable plotting.

        """
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.neuro_element = neuro_element
        self.rois, self.atlas = self._find_brain_atlas(self.neuro_element)

        self.hyperpipe_infos = None
        self.hyperpipe = hyperpipe
        self.hyperpipes_to_fit = {}

        self.roi_indices = {}
        self.create_surface_plots = create_surface_plots

    def _generate_mappings(self):
        """Generator creates dicts of hyperpipes by key roi_name."""
        for roi_index, roi_name in enumerate(self.rois):
            self.roi_indices[roi_name] = roi_index
            copy_of_hyperpipe = self.hyperpipe.copy_me()
            new_pipe_name = copy_of_hyperpipe.name + '_Atlas_Mapper_' + roi_name
            copy_of_hyperpipe.name = new_pipe_name
            copy_of_hyperpipe.output_settings.project_folder = self.folder
            copy_of_hyperpipe.output_settings.overwrite_results = True
            copy_of_hyperpipe.output_settings.save_output = True
            self.hyperpipes_to_fit[roi_name] = copy_of_hyperpipe
        return

    def fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None, **kwargs):
        """
        Transform data on NeuroElement and fit hyperpipes.
        :param X: input data
        :param y: targets
        :param kwargs:

        Returns:
            self

        """
        # disable fitting with loading from file/folder
        if not self.hyperpipes_to_fit and self.hyperpipe:
            self._generate_mappings()
        else:
            msg = "Cannot fit AtlasMapper with Hyperpipe as NoneType."
            logger.error(msg)
            raise ValueError(msg)

        # Get data from BrainAtlas first and save to .npz
        # ToDo: currently not supported for hyperparameters inside neurobranch
        self.neuro_element.fit(X)

        # extract regions
        X_extracted, _, _ = self.neuro_element.transform(X)
        X_extracted = AtlasMapper._reshape_roi_data(X_extracted)

        # save neuro element to file
        joblib.dump(self.neuro_element, os.path.join(self.folder, 'neuro_element.pkl'), compress=1)

        hyperpipe_infos = dict()
        hyperpipe_results = dict()

        # ToDo: parallel fitting
        for roi_name, hyperpipe in self.hyperpipes_to_fit.items():
            hyperpipe.verbosity = self.hyperpipe.verbosity
            hyperpipe.fit(X_extracted[self.roi_indices[roi_name]], y, **kwargs)
            hyperpipe_infos[roi_name] = {'hyperpipe_name': hyperpipe.name,
                                         'model_filename': os.path.join(os.path.basename(
                                             hyperpipe.output_settings.results_folder), 'photon_best_model.photon'),
                                         'roi_index': self.roi_indices[roi_name]}
            hyperpipe_results[roi_name] = ResultsHandler(hyperpipe.results).get_performance_outer_folds()

        self.hyperpipe_infos = hyperpipe_infos

        # write results
        with open(os.path.join(self.folder, self.hyperpipe.name + '_atlas_mapper_meta.json'), 'w') as fp:
            json.dump(self.hyperpipe_infos, fp)
        df = pd.DataFrame(hyperpipe_results)
        df.to_csv(os.path.join(self.folder, self.hyperpipe.name + '_atlas_mapper_results.csv'))

        # write performance to atlas niftis
        performances = list()

        for roi_name, roi_res in hyperpipe_results.items():
            n_voxels = len(X_extracted[self.roi_indices[roi_name]][0])
            performances.append(np.repeat(roi_res[self.hyperpipe.optimization.best_config_metric], n_voxels))

        backmapped_img, _, _ = self.neuro_element.inverse_transform(performances)
        backmapped_img.to_filename(os.path.join(self.folder, 'atlas_mapper_performances.nii.gz'))

        if self.create_surface_plots:
            self.surface_plots(backmapped_img)
        return self

    def predict(self, X: np.ndarray, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            msg = "No hyperpipes to predict. Did you remember to fit or load the AtlasMapper?"
            logger.error(msg)
            raise Exception(msg)

        X_extracted, _, _ = self.neuro_element.transform(X)
        X_extracted = AtlasMapper._reshape_roi_data(X_extracted)

        predictions = dict()
        for roi, infos in self.hyperpipe_infos.items():
            roi_index = infos['roi_index']
            predictions[roi] = self.hyperpipes_to_fit[roi].predict(X_extracted[roi_index], **kwargs)
        return predictions

    def surface_plots(self, perf_img):
        print('Creating surface plots')

        figure, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))
        axes = axes.ravel()
        big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

        cnt = 0
        for hemi, infl, sulc, pial in [('left', big_fsaverage.infl_left,
                                        big_fsaverage.sulc_left, big_fsaverage.pial_left),
                                       ('right', big_fsaverage.infl_right,
                                        big_fsaverage.sulc_right, big_fsaverage.pial_right)]:
            print('Hemi {}'.format(hemi))

            big_texture = surface.vol_to_surf(perf_img, pial, interpolation='nearest')

            for view in ['lateral', 'medial']:
                print('   View {}'.format(view))
                if cnt == 3:
                    output_file = os.path.join(self.folder, 'importance_scores_surface.png')
                else:
                    output_file = None
                plotting.plot_surf_stat_map(infl, big_texture, hemi=hemi, colorbar=True,
                                            title='{} hemisphere {} view'.format(hemi, view),
                                            threshold=0.0001, bg_map=sulc, view=view,
                                            output_file=output_file,
                                            axes=axes[cnt])
                cnt += 1

    @staticmethod
    def _find_brain_atlas(neuro_element: Union[NeuroBranch, PipelineElement]):
        """Find BrainAtlas and returns its rois and atlas_object."""
        roi_list = list()
        atlas_obj = list()
        if isinstance(neuro_element, NeuroBranch):
            for element in neuro_element.elements:
                if isinstance(element.base_element, BrainAtlas):
                    element.base_element.collection_mode = 'list'
                    roi_list, atlas_obj = AtlasMapper._find_rois(element)
        elif isinstance(neuro_element.base_element, BrainAtlas):
            neuro_element.base_element.collection_mode = 'list'
            roi_list, atlas_obj = AtlasMapper._find_rois(neuro_element)
        return roi_list, atlas_obj

    @staticmethod
    def _find_rois(element):
        roi_list = element.base_element.rois
        atlas_obj = AtlasLibrary().get_atlas(element.base_element.atlas_name)
        roi_objects = BrainAtlas._get_rois(atlas_obj, roi_list)
        return [roi.label for roi in roi_objects], atlas_obj

    @staticmethod
    def _reshape_roi_data(X):
        roi_data = [list() for _ in range(len(X[0]))]
        for roi_i in range(len(X[0])):
            for sub_i in range(len(X)):
                roi_data[roi_i].append(X[sub_i][roi_i])
        return roi_data

    @staticmethod
    def load_from_file(file: str):
        if not os.path.exists(file):
            msg = "Could not find the atlas-mapper meta file."
            logger.error(msg)
            raise FileNotFoundError(msg)
        return AtlasMapper._load(file)

    @staticmethod
    def load_from_folder(folder: str, analysis_name: str = None):
        if not os.path.exists(folder):
            raise NotADirectoryError("{} is not a directory".format(folder))

        if analysis_name:
            meta_file = glob(os.path.join(folder, analysis_name + '_atlas_mapper_meta.json'))
        else:
            meta_file = glob(os.path.join(folder, '*_atlas_mapper_meta.json'))

        if len(meta_file) == 0:
            raise FileNotFoundError("Couldn't find atlas_mapper_meta.json file in {}. "
                                    "Did you specify the correct analysis name?".format(folder))
        elif len(meta_file) > 1:
            raise ValueError("Found multiple atlas_mapper_meta.json files in {}".format(folder))

        return AtlasMapper._load(meta_file[0])

    @staticmethod
    def _load(file):
        # load neuro branch
        folder = os.path.split(file)[0]
        neuro_element = joblib.load(os.path.join(folder, 'neuro_element.pkl'))

        with open(file, "r") as read_file:
            hyperpipe_infos = json.load(read_file)

        roi_models = dict()
        for roi_name, infos in hyperpipe_infos.items():
            model_path = os.path.join(os.path.join(folder, infos['hyperpipe_name'] + "_results"),
                                      os.path.basename(infos['model_filename']))
            roi_models[roi_name] = Hyperpipe.load_optimum_pipe(model_path)
        atlas_mapper = AtlasMapper(neuro_element=neuro_element, folder=folder)
        atlas_mapper.hyperpipes_to_fit = roi_models
        atlas_mapper.hyperpipe_infos = hyperpipe_infos
        return atlas_mapper
