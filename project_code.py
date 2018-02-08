# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 05:47:02 2018

@author: Charlotte
"""

import numpy as np
import pandas as pd


sfile = 'C:\\Users\\Charlotte\\Documents\\Uni\\Physics\\Project\\Tachometer\\Data_capture\\capture3_fullrun_nostamp.txt'
tp = np.genfromtxt(sfile,dtype=np.str, delimiter=20)
print(tp)





df = pd.read_csv(sfile, sep=' ', header=None)
df.info()
print(df)



def hex_to_dex(strng_of_hex):
    return int(strng_of_hex, 16)
