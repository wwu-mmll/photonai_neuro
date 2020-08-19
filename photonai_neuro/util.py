import os
from photonai.base import PhotonRegistry


def delete_photonai_neuro():
    reg = PhotonRegistry()
    if 'photonai_neuro' in reg.PHOTON_REGISTRIES:
        reg.delete_module('photonai_neuro')


def register_photonai_neuro():
    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photonai_neuro.json")
    reg = PhotonRegistry()
    reg.add_module(current_path)
