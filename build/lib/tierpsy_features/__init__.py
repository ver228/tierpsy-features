# # -*- coding: utf-8 -*-
import os

EIGEN_PROJECTION_FILE = os.path.join(os.path.dirname(__file__), 'extras', 'master_eigen_worms_N2.mat')
assert os.path.exists(EIGEN_PROJECTION_FILE)