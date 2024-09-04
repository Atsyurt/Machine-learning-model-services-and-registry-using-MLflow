# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:41:37 2024

@author: ayhan
"""


import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'

# Verify that the environment variable is set
print(os.getenv('MLFLOW_TRACKING_URI'))
    