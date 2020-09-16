import unittest
from photonai_neuro.util import register_photonai_neuro, delete_photonai_neuro

from photonai.base.registry.registry import PhotonRegistry


class UtilTest(unittest.TestCase):

    def test_register_photonai_neuro(self):
        register_photonai_neuro()
        self.assertIn('photonai_neuro', PhotonRegistry.PHOTON_REGISTRIES)
        # multiple register should work
        register_photonai_neuro()
        self.assertIn('photonai_neuro', PhotonRegistry.PHOTON_REGISTRIES)

        delete_photonai_neuro()
        self.assertNotIn('photonai_neuro', PhotonRegistry.PHOTON_REGISTRIES)
        # multiple deletions
        delete_photonai_neuro()
        self.assertNotIn('photonai_neuro', PhotonRegistry.PHOTON_REGISTRIES)

        register_photonai_neuro()
        self.assertIn('photonai_neuro', PhotonRegistry.PHOTON_REGISTRIES)

        delete_photonai_neuro()
