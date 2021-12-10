import os
import numpy as np
from nibabel.nifti1 import Nifti1Image
from typing import Union

from photonai.base import ParallelBranch, CallbackElement, PhotonRegistry, PipelineElement
from photonai.photonlogger.logger import logger

from photonai_neuro.brain_atlas import BrainAtlas
from photonai_neuro.nifti_transformations import NeuroTransformerMixin


class NeuroBranch(ParallelBranch, NeuroTransformerMixin):
    """
    A substream of neuro elements that are encapsulated into a single block of PipelineElements to perform
    transformations on MRI data. A NeuroBranch takes niftis or nifti paths as input and should pass a numpy array
    to the subsequent PipelineElements.

    """
    NEURO_ELEMENTS = PhotonRegistry().get_package_info(['photonai_neuro'])

    def __init__(self, name: str, nr_of_processes: int = 1, output_img: bool = False):
        """
        Initialize the object

        Parameters:
            name:
                Name of the NeuroModule pipeline branch.

            nr_of_processes:
                Count of parallel processes.

            output_img:
                True -> return nifti object, False -> return np.ndarray (nifti.dataobj).

        """
        ParallelBranch.__init__(self, name, nr_of_processes=nr_of_processes)
        NeuroTransformerMixin.__init__(self, output_img=output_img)

    def __iadd__(self, pipe_element: PipelineElement):
        """
        Add an element to the neuro branch. Only neuro pipeline elements are allowed.

        Parameters:
            pipe_element:
                The transformer object to add. Should be registered in the neuro module or be a Callback.

        Returns:
            self

        """
        if pipe_element.name in NeuroBranch.NEURO_ELEMENTS:
            # as the neuro branch is parallelized and processes several images subsequently on
            # different cores, we need to stop the children to process on several cores as well
            pipe_element.base_element.output_img = True
            super(NeuroBranch, self).__iadd__(pipe_element)
        elif isinstance(pipe_element, CallbackElement):
            super(NeuroBranch, self).__iadd__(pipe_element)
        else:
            msg = 'PipelineElement {} is not part of the PHOTONAI Neuro module.'.format(pipe_element.name)
            logger.error(msg)
            raise ValueError(msg)
        return self

    def test_transform(self, X: np.ndarray, nr_of_tests: int = 1, save_to_folder: str = '.', **kwargs):
        """
        Apply a transform on a test subset for nr_of_tests samples.

        Parameters:
            X:
                The input samples as Niimg-like object of shape (n_samples, 1).

            nr_of_tests:
                Number of objects to be tested.

            save_to_folder:
                Path for saving the transformations.

            **kwargs:
                Passed to to the self.set_params.

        """
        if kwargs and len(kwargs) > 0:
            self.set_params(**kwargs)
        copy_of_me = self.copy_me()
        copy_of_me.nr_of_processes = 1
        copy_of_me.output_img = True
        # ToDo: What does this mean? extraction_mode cannot be 'list' for BrainAtlas
        # ToDo: Should test_transform work for brain_atlas?
        for p_element in copy_of_me.elements:
            if hasattr(p_element, 'base_element') and isinstance(p_element.base_element, BrainAtlas):
                p_element.base_element.extract_mode = 'list'

        filename = self.name + "_testcase_"

        new_pics, _, _ = copy_of_me.transform(X[:nr_of_tests])

        for i, new_pic in enumerate(new_pics):
            if not isinstance(new_pic, Nifti1Image):
                msg = "The test_transform function in NeuroBranch is only build for nifti file output. " \
                      "{} does not return a nifti image.".format(str(self.elements[-1]))
                logger.error(msg)
                raise ValueError(msg)

            new_filename = os.path.join(save_to_folder, filename + str(i) + "_transformed.nii")
            new_pic.to_filename(new_filename)

    def transform(self, X: np.ndarray, y: Union[None, np.ndarray] = None, **kwargs) -> (np.ndarray, np.ndarray, dict):
        """
        Apply transformation.

        Parameters:
            X:
                The input samples as Niimg-like object of shape (n_samples, 1).

            y:
                The input targets of shape (n_samples, 1).
                Not used in this transform.

            **kwargs:
                Passed to to the elements.

        Returns:
            Output of the last Branch-element.

        """
        X_new, y, kwargs = super(NeuroBranch, self).transform(X, y, **kwargs)

        # check if we have a list of niftis, should avoid this, except when output_image = True
        if not self.output_img:
            if ((isinstance(X_new, list) and len(X_new) > 0)
                or (isinstance(X_new, np.ndarray) and len(X_new.shape) == 1)) \
                    and isinstance(X_new[0], Nifti1Image):
                X_new = np.asarray([i.dataobj for i in X_new])
        return X_new, y, kwargs
