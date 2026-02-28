# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain — Local-First Anonymous RAG Knowledge Subnet
# MIT License

__version__ = "1.0.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

from . import protocol
from . import base
from . import validator
from .subnet_links import SUBNET_LINKS
