import json
import os
from glob import glob
from typing import Union
import warnings

import joblib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from nilearn import datasets, surface, plotting

from photonai.base import PipelineElement, Stack, Branch
from photonai.base.hyperpipe import Hyperpipe
from photonai.photonlogger.logger import logger
from photonai.processing import ResultsHandler

from photonai_neuro.brain_atlas import BrainAtlas, AtlasLibrary
from photonai_neuro.neuro_branch import NeuroBranch


class AtlasStacker:
    """
    Mapping between neuro processing and Hyperpipe containing Stack with Branches for regions of interest.

    Parameter
    ---------
    * `neuro_element` [Union[NeuroBranch, PipelineElement]]:
        Neuro processing in front of hyperpipe.

    * `hyperpipe` [photonai.base.Hyperpipe]
        Hyperpipe to fit without any PipelineElement.

    * `roi_branch` [photonai.base.Branch]:
        Branch calculated on every ROI.

    * `final_estimator` [photonai.base.PipelineElement]:
        Final estimator.

    """

    def __init__(self,
                 neuro_element: Union[NeuroBranch, PipelineElement],
                 hyperpipe: Hyperpipe = None,
                 roi_branch: Branch = None,
                 folder: str = "/.tmp",
                 final_estimator: PipelineElement = None):

        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.neuro_element = neuro_element
        self.rois, self.brain_atlas = self._find_brain_atlas(self.neuro_element)

        self.hyperpipe_infos = None
        self.hyperpipe = hyperpipe
        if hyperpipe:
            if hyperpipe.elements:
                msg = "AtlasStacker needs an element-empty Hyperpipe." \
                      "Please give the elements into the roi_branch parameter."
                logger.error(msg)
                raise ValueError(msg)
            self.hyperpipe.output_settings.project_folder = self.folder
            self.hyperpipe.output_settings.overwrite_results = True
            self.hyperpipe.output_settings.save_output = True

        self.roi_indices = {}
        self.roi_branch = roi_branch
        self.final_estimator = final_estimator


    def _generate_mappings(self):
        """
        Generator creates hyperpipe.
        :return: _
        """
        self.rois, self.brain_atlas = self._find_brain_atlas(self.neuro_element)
        stack = Stack("RoiStack")
        for roi_index, roi_name in enumerate(self.rois):
            self.roi_indices[roi_name] = roi_index
            branch = Branch(roi_name + "_branch")
            branch += PipelineElement("RoiFilter",
                                      roi_allocation=self.brain_atlas.roi_allocation,
                                      rois=roi_name,
                                      mask_indices=self.brain_atlas.mask_indices)
            for el in self.roi_branch.elements:
                branch += el.copy_me()
            stack += branch

        self.hyperpipe += stack
        self.hyperpipe += self.final_estimator
        return

    def fit(self, X, y=None, **kwargs):
        """
        Transform data on NeuroElement and fit hyperpipes.
        :param X: input data
        :param y: targets
        :param kwargs:
        :return:
        """

        # disable fitting with loading from file/folder
        if np.any([self.hyperpipe, self.final_estimator, self.roi_branch]) is None:
            msg = "Cannot fit AtlasStacker with one of [hyperpipe, roi_branch, final_estimator] as NoneType."
            logger.error(msg)
            raise ValueError(msg)

        # ToDo: currently not supported for hyperparameters inside neurobranch
        self.neuro_element.fit(X)

        # extract regions
        X_extracted, _, _ = self.neuro_element.transform(X)

        # save neuro branch to file
        joblib.dump(self.neuro_element, os.path.join(self.folder, 'neuro_element.pkl'), compress=1)

        self._generate_mappings()
        self.hyperpipe.fit(X_extracted, y, **kwargs)


    def predict(self, X, **kwargs):
        if self.hyperpipe is None:
            msg = "No hyperpipe to predict. Did you remember to load the Atlas Mapper?"
            logger.error(msg)
            raise Exception(msg)

        X_extracted, _, _ = self.neuro_element.transform(X)

        return self.hyperpipe.predict(X_extracted, **kwargs)

    @staticmethod
    def _find_brain_atlas(neuro_element: Union[NeuroBranch, PipelineElement]):
        """
        Find BrainAtlas and returns its rois and atlas_object.
        :param neuro_element: NeuroElement
        :return: (roi_list, atlas_obj)
        """
        roi_list = list()
        if isinstance(neuro_element, NeuroBranch):
            for element in neuro_element.elements:
                if isinstance(element.base_element, BrainAtlas):
                    roi_list, atlas_obj = AtlasStacker._find_rois(element)
                    brain_atlas = element.base_element
        elif isinstance(neuro_element.base_element, BrainAtlas):
            roi_list, atlas_obj = AtlasStacker._find_rois(neuro_element)
            brain_atlas = neuro_element.base_element
        return roi_list, brain_atlas

    @staticmethod
    def _find_rois(element):
        roi_list = element.base_element.rois
        atlas_obj = AtlasLibrary().get_atlas(element.base_element.atlas_name)
        roi_objects = BrainAtlas._get_rois(atlas_obj, roi_list)
        return [roi.label for roi in roi_objects], atlas_obj

    @staticmethod
    def load_from_file(file: str):
        if not os.path.exists(file):
            raise FileNotFoundError("Couldn't find atlas mapper meta file")

        return AtlasStacker._load(file)

    @staticmethod
    def load_from_folder(folder: str, hyperpipe_name: str = None):
        if not os.path.exists(folder):
            raise NotADirectoryError("{} is not a directory".format(folder))

        if hyperpipe_name:
            best_model = glob(os.path.join(folder, hyperpipe_name + '_results'+'.json'))
        else:
            best_model = glob(os.path.join(folder, '*_atlas_mapper_meta.json'))

        if len(meta_file) == 0:
            raise FileNotFoundError("Couldn't find atlas_mapper_meta.json file in {}. Did you specify the correct analysis name?".format(folder))
        elif len(meta_file) > 1:
            raise ValueError("Found multiple atlas_mapper_meta.json files in {}".format(folder))

        return AtlasStacker._load(best_model)

    @staticmethod
    def _load(file):
        # load neuro branch
        folder = os.path.split(file)[0]
        neuro_element = joblib.load(os.path.join(folder, 'neuro_element.pkl'))
        trained_hyperpipe = Hyperpipe.load_optimum_pipe(file)
        atlas_mapper = AtlasStacker(neuro_element=neuro_element, folder=folder)
        atlas_mapper.hyperpipe = trained_hyperpipe
        return atlas_mapper
