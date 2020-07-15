import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_img, index_img, smooth_img

from photonai.base import PipelineElement

from photonai_neuro import NeuroBranch
from test.test_neuro import NeuroBaseTest


class SmoothImagesTests(NeuroBaseTest):

    def test_single_subject_smoothing(self):

        # nilearn
        nilearn_smoothed_img = smooth_img(self.X[0], fwhm=[3, 3, 3])
        nilearn_smoothed_array = nilearn_smoothed_img.dataobj

        # photon
        smoother = PipelineElement('SmoothImages', hyperparameters={}, fwhm=3, batch_size=1)
        photon_smoothed_array, _, _ = smoother.transform(self.X[0])

        branch = NeuroBranch('NeuroBranch', output_img=True)
        branch += smoother
        photon_smoothed_img, _, _ = branch.transform(self.X[0])

        # assert
        self.assertIsInstance(photon_smoothed_array, np.ndarray)
        self.assertIsInstance(photon_smoothed_img, Nifti1Image)

        self.assertTrue(np.array_equal(photon_smoothed_array, nilearn_smoothed_array))
        self.assertTrue(np.array_equal(photon_smoothed_img.dataobj, nilearn_smoothed_img.dataobj))

    def test_multi_subject_smoothing(self):
        # nilearn
        nilearn_smoothed_img = smooth_img(self.X[0:3], fwhm=[3, 3, 3])
        nilearn_smoothed_array = nilearn_smoothed_img[1].dataobj

        # photon
        smoother = PipelineElement('SmoothImages', hyperparameters={}, fwhm=3)
        photon_smoothed_array, _, _ = smoother.transform(self.X[0:3])

        branch = NeuroBranch('NeuroBranch', output_img=True)
        branch += smoother
        photon_smoothed_img, _, _ = branch.transform(self.X[0:3])

        # assert
        self.assertIsInstance(photon_smoothed_array, np.ndarray)
        self.assertIsInstance(photon_smoothed_img[0], Nifti1Image)

        self.assertTrue(np.array_equal(photon_smoothed_array[1], nilearn_smoothed_array))
        self.assertTrue(np.array_equal(photon_smoothed_img[1].dataobj, nilearn_smoothed_img[1].dataobj))


class ResampleImagesTests(NeuroBaseTest):

    def test_single_subject_resampling(self):
        voxel_size = [3, 3, 3]

        # nilearn
        nilearn_resampled_img = resample_img(self.X[0], interpolation='nearest', target_affine = np.diag(voxel_size))
        nilearn_resampled_array = nilearn_resampled_img.dataobj

        # photon
        resampler = PipelineElement('ResampleImages', hyperparameters={}, voxel_size=voxel_size, batch_size=1)
        single_resampled_img, _, _ = resampler.transform(self.X[0])

        branch = NeuroBranch('NeuroBranch', output_img=True)
        branch += resampler
        branch_resampled_img, _, _ = branch.transform(self.X[0])

        # assert
        self.assertIsInstance(single_resampled_img, np.ndarray)
        self.assertIsInstance(branch_resampled_img[0], Nifti1Image)

        self.assertTrue(np.array_equal(nilearn_resampled_array, single_resampled_img))
        self.assertTrue(np.array_equal(single_resampled_img, branch_resampled_img[0].dataobj))

    def test_multi_subject_resampling(self):
        voxel_size = [3, 3, 3]

        # nilearn
        nilearn_resampled = resample_img(self.X[:3], interpolation='nearest', target_affine = np.diag(voxel_size))
        nilearn_resampled_img = [index_img(nilearn_resampled, i) for i in range(nilearn_resampled.shape[-1])]
        nilearn_resampled_array = np.moveaxis(nilearn_resampled.dataobj, -1, 0)

        # photon
        resampler = PipelineElement('ResampleImages', hyperparameters={}, voxel_size=voxel_size)
        resampled_img, _, _ = resampler.transform(self.X[:3])

        branch = NeuroBranch('NeuroBranch', output_img=True)
        branch += resampler
        branch_resampled_img, _, _ = branch.transform(self.X[:3])

        # assert
        self.assertIsInstance(resampled_img, np.ndarray)
        self.assertIsInstance(branch_resampled_img, list)
        self.assertIsInstance(branch_resampled_img[0], Nifti1Image)

        self.assertTrue(np.array_equal(nilearn_resampled_array, resampled_img))
        self.assertTrue(np.array_equal(branch_resampled_img[1].dataobj, nilearn_resampled_img[1].dataobj))


class PatchImagesTests(NeuroBaseTest):

    def setUp(self) -> None:
        pass