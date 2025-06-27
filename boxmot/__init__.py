# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '12.0.7'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.imprassoc.imprassoctrack import ImprAssocTrack


TRACKERS = ['imprassoc']

__all__ = ("__version__",
           "StrongSort", "OcSort", "ByteTrack", "BotSort", "DeepOcSort", "HybridSort", "ImprAssocTrack", "BoostTrack",
           "create_tracker", "get_tracker_config", "gsi")
