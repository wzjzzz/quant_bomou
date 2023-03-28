# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:26:30 2022

@author: asd
"""

import pandas as pd
import os
import sys

pwd_input, pwd_output = sys.argv[1], sys.argv[2]
for filename in os.listdir(pwd_input):
    url = os.path.join(pwd_input, filename)
    data = pd.read_csv(url)
    data.drop(data.columns[4], axis=1, inplace=True)
    url = os.path.join(pwd_output, filename)
    data.to_csv(url)
