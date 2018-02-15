#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:32:54 2018

@author: edwardtaylor
"""


import numpy as np
from itertools import accumulate
import os
import numpy as np
import matplotlib.pyplot as plt

sfile = '/Users/edwardtaylor/Desktop/data.txt'
tp = np.genfromtxt(sfile,dtype=np.str, delimiter=20)

def splitupbyte(bytess):
    res = []
    for b in bytess:
        pairs = []
        for n in range(len(b)//2):
            pairs.append(b[2*n:2*n+2])
        res.append(np.array(pairs))
    return np.array(res)

def filter_data(data):
    ret = []
    for element in data:
        w = list(element)
        for m in [0,1,2,2,5]:
            g = w.pop(m)
        ret.append(np.array(w))
    return np.array(ret)

def interpret(data):
    ret = []
    for element in data:
        # element has size 5
        # Convert subcomponents 2,3,4
        w = list(element)
        for m in [2,3,4]:
            w[m] = hex_to_dex(w[m])
        ret.append(w)
    return np.array(ret)

def hex_to_dex(h):
    return int(h, 16)

def reading(data):
    readings = []
    for k in data:
        value = float(k[2]) + 256*float(k[3]) + 65536*float(k[4])
        readings.append((value/(10**(int(k[0])/2 + 1)), int(k[1])))
    return readings

def get_time(data):
    readings=[]
    for i in data:
        readings.append((float(60/i[0])))
    output = list(accumulate(readings))
    return output
         
def main():
    data = splitupbyte(tp)
    filtered_data = filter_data(data)
    convert = interpret(filtered_data)
    readout = reading(convert)
    velocity = []
    for i in readout:
        velocity.append(float(i[0]))
    print(velocity)
    time = get_time(reading(interpret(filter_data(splitupbyte(tp)))))
    print(time)
    
    '''    if i[1] == 1:
            unit = 'rpm'
        elif i[1] == 2:
            unit = 'm/min'
        elif i[1] == 3:
            unit = 'ft/min'
        elif i[1] == 4:
            unit = 'yd/min'
        elif i[1] == 5:
            unit = 'rps'
        else:
            unit = ''
        print(str(i[0]) + ' {:}'.format(unit))'''
    fig = plt.figure(num=1, figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time/s")
    ax.set_ylabel("Rotations per minute/rpm")
    ax.set_title("Stuff")
    ax.plot(velocity, time, 'b.', label='data')
    ax.legend()
main()




