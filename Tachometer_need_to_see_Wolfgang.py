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
    return list(accumulate(readings))


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def func(x, w_0, b, c):
    return w_0*np.exp(-b*x)+c
    
def ply_func(x,a,b,c,d,e,f):
    return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f

def c_rsd(real_function,time,a,b,c):
    fitted_data = func(time,a,b,c)
    return np.array(real_function)-fitted_data

def main():
    data = splitupbyte(tp)
    filtered_data = filter_data(data)
    convert = interpret(filtered_data)
    readout = reading(convert)
    v = []
    for i in readout:
        v.append(float(i[0]))
    t = np.array(get_time(reading(interpret(filter_data(splitupbyte(tp))))))
    
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
    p, pc = optimize.curve_fit(func,t,v)
    sd = np.sqrt(np.diag(pc))
    fig = plt.figure(num=1, figsize=(10, 20))
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel("Time/s", fontsize=16)
    ax.set_ylabel("Rotations per minute/rpm", fontsize=14)
    ax.set_title("Rotations vs Time", fontsize=14)
    
    ax.plot(t,v,'b.',label='data')
    ax.plot(t, func(t,p[0],p[1],p[2]),'r.',label='Fitted function')
    ax.text(0.00, 60.00, 'w_0 = '+str(p[0])+'±'+ str(sd[0])
    +'\n'+'a = '+str(p[1])+'±'+ str(sd[1])
    +'\n'+'b = '+str(p[2])+'±'+ str(sd[2]))
    
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t,c_rsd(v,t,p[0],p[1],p[2]),'r')
    ax2.set_xlabel("Time/s", fontsize=16)
    ax2.set_ylabel("Residuals/rpm", fontsize=14)
    ax2.set_title("Residuals", fontsize=14)
    ax2.axhline(0.)
    
    p2, pc2 = optimize.curve_fit(ply_func,t,c_rsd(v,t,p[0],p[1],p[2]))
    sd2 = np.sqrt(np.diag(pc2))
    ax2.plot(t,ply_func(t,p2[0],p2[1],p2[2],p2[3],p2[4],p2[5]),'g')
    ax2.text(20.00, 1.50,'a = '+str(p2[0])+'±'+str(sd2[0]) 
    +'\n'+'b = '+str(p2[1])+'±'+str(sd2[1]) 
    +'\n'+'c = '+str(p2[2])+'±'+str(sd2[2])
    +'\n'+'d = '+str(p2[3])+'±'+str(sd2[3])
    +'\n'+'e = '+str(p2[4])+'±'+str(sd2[4]))
    
    ax.legend()
   
    plt.savefig('Figure2.jpg')
    
main()