#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:32:54 2018

@author: edwardtaylor
"""


import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.gridspec as gridspec 
I = 4*(10**-3)

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

def func(x, w, a, b):
    return w*np.exp(-a*x)+b
 
def create_residuals(velocity,time,w,a,b):
    fitted_data = func(np.array(time),w ,a, b)
    return np.array(velocity)-fitted_data

def main():
    data = splitupbyte(tp)
    filtered_data = filter_data(data)
    convert = interpret(filtered_data)
    readout = reading(convert)
    velocity = []
    for i in readout:
        velocity.append(float(i[0]))
    time = get_time(reading(interpret(filter_data(splitupbyte(tp)))))
    
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
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])    
    params, params_covariance = optimize.curve_fit(func, time, velocity)
    fig = plt.figure(num=1, figsize=(16, 12))
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel("Time/s", fontsize=16)
    ax.set_ylabel("Rotations per minute/rpm", fontsize=14)
    ax.set_title("Rotations vs Time", fontsize=14)
    ax.plot(time, velocity, 'b.', label='data')
    ax.plot(time, func(np.array(time), params[0], params[1], params[2]), 'r.', label='Fitted function')
    ax.text(0.00, 60.00, 'w = ' + str(params[0]) + '\n' + 'a = ' + str(params[1]) + '\n' + 'b = ' + str(params[2]))
    ax = fig.add_subplot(gs[1])
    ax.plot(time,create_residuals(velocity,time,params[0],params[1],params[2]),'r')
    ax.set_xlabel("Time/s", fontsize=16)
    ax.set_ylabel("Residuals/rpm", fontsize=14)
    ax.set_title("Residuals", fontsize=14)
    ax.legend()
    
    plt.savefig('Figure2.jpg')
    
main()
