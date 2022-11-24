import os
from photonai.base import PhotonRegistry
from photonai.photonlogger import logger

from .util import register_photonai_neuro
from .version import __version__

from .atlas_mapper import AtlasMapper
from .brain_mask import BrainMask
from .brain_atlas import BrainAtlas
from .atlas_library import AtlasLibrary
from .neuro_branch import NeuroBranch


# REGISTRATION
def do_register(current_path, registered_file):  # pragma: no cover
    reg = PhotonRegistry()
    reg.add_module(os.path.join(current_path, "photonai_neuro.json"))
    with open(os.path.join(registered_file), "w") as f:
        f.write(str(__version__))


def register():  # pragma: no cover
    current_path = os.path.dirname(os.path.abspath(__file__))
    registered_file = os.path.join(current_path, "registered")
    logger.info("Checking Neuro Module Registration")
    if not os.path.isfile(registered_file):  # pragma: no cover
        logger.info("Registering Neuro Module")
        do_register(current_path=current_path, registered_file=registered_file)
    else:
        with open(os.path.join(registered_file), "r") as f:
            if f.read() == __version__:
                logger.info("Current version already registered")
            else:
                logger.info("Updating Neuro Module")
                do_register(current_path=current_path, registered_file=registered_file)


register()
