import os
from datetime import datetime

from photonai.photonlogger import logger

from .util import register_photonai_neuro


# REGISTRATION
current_path = os.path.dirname(os.path.abspath(__file__))
registered_file = os.path.join(current_path, "registered")
logger.info("Checking Neuro Module Registration")
if not os.path.isfile(registered_file):
    logger.info("Registering Neuro Module")
    register_photonai_neuro()
    with open(os.path.join(registered_file), "w") as f:
        f.write(str(datetime.now()))

from .atlas_mapper import AtlasMapper
from .brain_mask import BrainMask
from .brain_atlas import BrainAtlas
from .atlas_library import AtlasLibrary
from .neuro_branch import NeuroBranch
