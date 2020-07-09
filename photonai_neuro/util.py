from photonai.base import PhotonRegistry
import os

def delete_photonai_neuro():
    reg = PhotonRegistry()
    reg.delete_module('photonai_neuro')


def register_photonai_neuro():
    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photonai_neuro/photonai_neuro.json")
    reg = PhotonRegistry()
    reg.add_module(current_path)
