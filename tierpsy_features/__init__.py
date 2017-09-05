# # -*- coding: utf-8 -*-
import os

EIGEN_PROJECTION_FILE = os.path.join(os.path.dirname(__file__), 'extras', 'master_eigen_worms_N2.mat')
assert os.path.exists(EIGEN_PROJECTION_FILE)

from .velocities import get_velocity_features
from .postures import get_morphology_features, get_posture_features
from .smooth import SmoothedWorm