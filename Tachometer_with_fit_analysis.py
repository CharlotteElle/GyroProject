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
 
def poly_func(x,a,b,c,d,e,f):
    return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f

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
    gs = gridspec.GridSpec(3, 1)    
    params, pcov = optimize.curve_fit(func, time, velocity)
    std_dev = np.sqrt(np.diag(pcov))
    fig = plt.figure(num=1, figsize=(16, 12))
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel("Time/s", fontsize=16)
    ax.set_ylabel("Rotations per minute/rpm", fontsize=14)
    ax.set_title("Rotations vs Time", fontsize=14)
    
    ax.plot(time, velocity, 'b.', label='data')
    ax.plot(time, func(np.array(time), params[0], params[1], params[2]), 'r.', label='Fitted function')
    ax.text(0.00, 60.00, 'w = ' + str(params[0])+'±'+ str(std_dev[0]) + '\n' + 'a = ' + str(params[1])+'±'+ str(std_dev[1]) + '\n' + 'b = ' + str(params[2])+'±'+ str(std_dev[2]))
    ax.legend()
    
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time,create_residuals(velocity,time,params[0],params[1],params[2]),'r')
    ax2.set_xlabel("Time/s", fontsize=16)
    ax2.set_ylabel("Residuals/rpm", fontsize=14)
    ax2.set_title("Residuals", fontsize=14)
    ax2.axhline(0.)
    
    params2, pcov2 = optimize.curve_fit(poly_func,time,create_residuals(velocity,time,params[0],params[1],params[2]))
    std_dev2 = np.sqrt(np.diag(pcov2))
    ax2.plot(time,poly_func(np.array(time),params2[0],params2[1],params2[2],params2[3],params2[4],params2[5]),'g')
    ax2.text(20.00, 1.50, 'a = ' + str(params2[0])+'±'+ str(std_dev2[0]) + '\n' + 'b = ' + str(params2[1])+'±'+ str(std_dev2[1]) 
+'\n'+'c = '+str(params2[2])+'±'+ str(std_dev2[2])+'\n' + 'd = '+str(params2[3])+'±'+ str(std_dev2[3])+'\n'+'e = '+str(params2[4])+'±'+ str(std_dev2[4]))
    ax2.legend()
    
    plt.savefig('Figure2.jpg')
    
main()
