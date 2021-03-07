import glob
import os
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn import image

from photonai.base import PipelineElement, CallbackElement
from photonai.base.photon_pipeline import CacheManager

from photonai_neuro import NeuroBranch
from test.test_neuro import NeuroBaseTest


class NeuroBranchTests(NeuroBaseTest):

    def test_resampling_and_smoothing(self):
        """
        Procedure:
            1. Testing Method on Single Core
            2. Testing Method on Multi Core
            3. Testing Method on Single Core Batched
            4. Testing Method on Multi Core Batched

        """
        def create_instances_and_transform(neuro_class_str, param_dict, transformed_X):

            for i in range(4):
                if i in [0, 2]:
                    obj = NeuroBranch(name="single core application", nr_of_processes=1)
                else:
                    obj = NeuroBranch(name="multi core application", nr_of_processes=3)

                if i in [0, 1]:
                    obj += PipelineElement(neuro_class_str, **param_dict)
                else:
                    obj += PipelineElement(neuro_class_str, batch_size=5, **param_dict)

                # transform data
                obj.base_element.cache_folder = self.cache_folder_path
                CacheManager.clear_cache_files(obj.base_element.cache_folder, True)
                obj.base_element.current_config = {'test_suite': 1}

                new_X, _, _ = obj.transform(self.X)
                obj.base_element.clear_cache()

                # compare output to nilearn version
                for index, nilearn_nifti in enumerate(transformed_X):
                    photon_nifti = new_X[index]
                    if isinstance(photon_nifti, Nifti1Image):
                        self.assertTrue(np.array_equal(photon_nifti.dataobj, nilearn_nifti.dataobj))
                    else:
                        self.assertTrue(np.array_equal(np.asarray(photon_nifti), nilearn_nifti.dataobj))

                print("finished testing object: all images are fine.")

        print("Testing Nifti Smoothing.")
        smoothing_param_dict = {'fwhm': [3, 3, 3]}
        nilearn_smoothed_X = []
        for element in self.X:
            nilearn_smoothed_X.append(image.smooth_img(element, **smoothing_param_dict))
        create_instances_and_transform('SmoothImages', smoothing_param_dict, nilearn_smoothed_X)

        print("Testing Nifti Resampling.")
        target_affine = np.diag([5, 5, 5])
        resample_param_dict = {'target_affine': target_affine, 'interpolation': 'nearest'}
        nilearn_resampled_X = []
        for element in self.X:
            nilearn_resampled_X.append(image.resample_img(element, **resample_param_dict))
        create_instances_and_transform('ResampleImages', {'voxel_size': [5, 5, 5]}, nilearn_resampled_X)

    def test_neuro_module_branch(self):
        nmb = NeuroBranch('best_branch_ever')
        nmb += PipelineElement('SmoothImages', fwhm=10)
        nmb += PipelineElement('ResampleImages', voxel_size=5)
        nmb += PipelineElement('BrainAtlas', rois=['Hippocampus_L', 'Hippocampus_R'],
                               atlas_name="AAL", extract_mode='vec')

        nmb.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nmb.base_element.cache_folder, True)
        # set the config so that caching works
        nmb.set_params(**{'SmoothImages__fwhm': 10, 'ResampleImages__voxel_size': 5})

        # transforming 8 Niftis with 3 elements, so afterwards there should be 3*8
        nr_niftis = 7
        nmb.transform(self.X[:nr_niftis])
        nr_files_in_folder = len(glob.glob(os.path.join(nmb.base_element.cache_folder, "*.p")))
        self.assertTrue(nr_files_in_folder == 3 * nr_niftis)
        self.assertTrue(len(nmb.base_element.cache_man.cache_index.items()) == (3*nr_niftis))

        # transform 3 items that should have been cached and two more that need new processing
        nmb.transform(self.X[nr_niftis-2::])
        # now we should have 10 * 3
        nr_files_in_folder = len(glob.glob(os.path.join(nmb.base_element.cache_folder, "*.p")))
        self.assertTrue(nr_files_in_folder == (3 * len(self.X)))
        self.assertTrue(len(nmb.base_element.cache_man.cache_index.items()) == (3 * len(self.X)))

    def test_output_img(self):
        for output_img in [True, False]:
            nb = NeuroBranch('Neuro_Branch', output_img=output_img)
            nb += PipelineElement('SmoothImages', fwhm=10)
            nb += PipelineElement('ResampleImages', voxel_size=5)

            nb.base_element.cache_folder = self.cache_folder_path
            CacheManager.clear_cache_files(nb.base_element.cache_folder, True)
            # set the config so that caching works
            nb.set_params(**{'SmoothImages__fwhm': 10, 'ResampleImages__voxel_size': 5})

            results, _, _ = nb.transform(self.X[:6])

            if output_img:
                for res in results:
                    self.assertIsInstance(res, Nifti1Image)
            else:
                self.assertEqual(len(results.shape), 4)
                self.assertEqual(results.shape[0], 6)

    def test_test_transform_single(self):
        nb = NeuroBranch('neuro_branch')
        nb += PipelineElement('SmoothImages', fwhm=10)
        nb += PipelineElement('ResampleImages', voxel_size=5)

        nb.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nb.base_element.cache_folder, True)
        # set the config so that caching works
        nb.set_params(**{'SmoothImages__fwhm': 10, 'ResampleImages__voxel_size': 5})

        nb.test_transform(self.X)

        self.assertTrue(os.path.exists("./neuro_branch_testcase_0_transformed.nii"))
        os.remove("./neuro_branch_testcase_0_transformed.nii")

        with self.assertRaises(ValueError):
            nb += PipelineElement('BrainAtlas', rois=['Hippocampus_L', 'Hippocampus_R'],
                               atlas_name="AAL", extract_mode='vec')
            nb.test_transform(self.X)

    def test_test_transform_multi(self):
        nb = NeuroBranch('neuro_branch', nr_of_processes=2)
        nb += PipelineElement('SmoothImages', fwhm=10)
        nb += PipelineElement('ResampleImages', voxel_size=5)

        nb.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nb.base_element.cache_folder, True)
        # set the config so that caching works
        nb.set_params(**{'SmoothImages__fwhm': 10, 'ResampleImages__voxel_size': 5})

        nb.test_transform(self.X, nr_of_tests=3)

        for i in range(3):
            self.assertTrue(os.path.exists("./neuro_branch_testcase_{}_transformed.nii".format(str(i))))
            os.remove("./neuro_branch_testcase_{}_transformed.nii".format(str(i)))

    def test_test_transform_maskatlas_error(self):
        nb = NeuroBranch('neuro_branch')
        nb += PipelineElement('SmoothImages', fwhm=10)
        nb += PipelineElement('ResampleImages', voxel_size=5)
        custom_mask = os.path.join(self.atlas_folder, 'Cerebellum/P_08_Cere.nii.gz')
        nb += PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='vec')

        nb.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nb.base_element.cache_folder, True)
        # set the config so that caching works

        # last element is not returning in nii-format
        with self.assertRaises(ValueError):
            nb.test_transform(self.X)

    def test_core_element_error(self):
        nb = NeuroBranch('neuro_branch')
        with self.assertRaises(ValueError):
            nb += PipelineElement('SVC')

    def test_callback(self):
        self.a = None
        def my_monitor(X, y=None, **kwargs):
            self.a = X

        nb = NeuroBranch('neuro_branch')
        nb += PipelineElement('SmoothImages', fwhm=10)
        nb += PipelineElement('ResampleImages', voxel_size=5)
        nb += CallbackElement("monitor", my_monitor)

        nb.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nb.base_element.cache_folder, True)
        # set the config so that caching works
        nb.set_params(**{'SmoothImages__fwhm': 10, 'ResampleImages__voxel_size': 5})

        nb.transform(self.X[:1])

        self.assertIsInstance(self.a[0], Nifti1Image)

    def test_invariance(self):
        nb1 = NeuroBranch('neuro_branch', nr_of_processes=4, output_img=True)
        nb1 += PipelineElement('ResampleImages', voxel_size=5, batch_size=1, interpolation='nearest')
        nb1.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nb1.base_element.cache_folder, True)
        pai_result1, _, _ = nb1.transform(self.X)
        del nb1

        nb2 = NeuroBranch('neuro_branch', nr_of_processes=4, output_img=False)
        nb2 += PipelineElement('ResampleImages', voxel_size=5, batch_size=2, interpolation='nearest')
        nb2.base_element.cache_folder = self.cache_folder_path
        CacheManager.clear_cache_files(nb2.base_element.cache_folder, True)
        pai_result2, _, _ = nb2.transform(self.X)

        nilearn_result = image.resample_img(self.X,
                                            target_affine=np.diag([5, 5, 5]),
                                            interpolation='nearest')
        nilearn_result = [image.index_img(nilearn_result, i) for i in range(nilearn_result.shape[-1])]

        for i in range(len(self.X)):
            np.testing.assert_equal(nilearn_result[i].dataobj, pai_result2[i, :, :, :])
            np.testing.assert_equal(nilearn_result[i].dataobj, pai_result1[i].dataobj)
