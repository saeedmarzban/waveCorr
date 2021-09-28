# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:46:35 2021

@author: Anonymous
"""

from enum import Enum

class normalization_mode(Enum):
    no_norm = 0
    standard_norm = 1
    capped_norm = 2
    
class activation_functions(Enum):
    relu = 0
    sigmoid = 1
    tanh = 1
    linear = 2