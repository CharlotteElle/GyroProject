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
from scipy import stats
import matplotlib.gridspec as gridspec 
import sympy

I = 4*10**-3
x,t,v,w,b,c,d = sympy.symbols('x t v w b c d')
sfile = '/Users/edwardtaylor/Desktop/data2.txt'
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

def func(x, w_0, b, c,d):
    return (w_0-d)*np.exp(-b*x)+c*x+d
    
def fit_func(x,a,b,c,d):
    return a*np.sin(b*x+c)+d

def resid_func(real_function,time,a,b,c,d):
    fitted_data = func(time,a,b,c,d)
    return np.array(real_function)-fitted_data

def avg_fn(f):
    ret = []
    for i in range(0,len(f)):
        ret.append(np.average(np.array([(f[i:i+40])])))
    ret = ret[15:-40]
    return ret

def improv_func(real_function,time,a,b,c,d):
    fitted_data = fit_func(time,a,b,c,d)
    return np.array(real_function)+fitted_data

def main():
    data = splitupbyte(tp)
    filtered_data = filter_data(data)
    convert = interpret(filtered_data)
    readout = reading(convert)
    v = []
    for i in readout:
        v.append(float(i[0]))
    t = np.array(get_time(reading(interpret(filter_data(splitupbyte(tp))))))
    tt = t[15:-40]
    
    gs = gridspec.GridSpec(4, 1)    
    p, pc = optimize.curve_fit(func,t,v)
    sd = np.sqrt(np.diag(pc))
    fig = plt.figure(num=1, figsize=(30, 30))
    ax = fig.add_subplot(gs[0])
    ax.set_xlabel("Time/s", fontsize=16)
    ax.set_ylabel("Rotations per minute/rpm", fontsize=14)
    ax.set_title("Rotations vs Time", fontsize=14)
    
    ax.plot(t,v,'b.',label='data')
    ax.plot(t, func(t,p[0],p[1],p[2],p[3]),'r.',label='Fitted function')
    ax.errorbar(t,v,xerr=0.001,yerr=0.01,label='Error bars')
    r = stats.linregress(v,func(t,p[0],p[1],p[2],p[3]))[2]
    chi_sqr = stats.chisquare(v,func(t,p[0],p[1],p[2],p[3]))[0]
    
    ax.text(200.00, 150.00, 'w_0 = '+str(p[0])+'±'+ str(sd[0])
    +'\n'+'b = '+str(p[1])+'±'+str(sd[1])
    +'\n'+'c = '+str(p[2])+'±'+str(sd[2])
    +'\n'+'d = '+str(p[3])+'±'+str(sd[3])
    +'\n'+'r-squared = '  +str(r**2)
    +'\n'+'chi-squared = '+str(chi_sqr)
    )
    ax.legend()
    
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t,resid_func(v,t,p[0],p[1],p[2],p[3]),'r',label='Residuals')
    ax2.set_xlabel("Time/s", fontsize=16)
    ax2.set_ylabel("Residuals/rpm", fontsize=14)
    ax2.set_title("Residuals", fontsize=14)
    ax2.axhline(0.,label = 'Zero Line')
    ax2.errorbar(tt,avg_fn(resid_func(v,t,p[0],p[1],p[2],p[3])),xerr=0.01,yerr=0.4,label='Error bars')
    ax2.plot(tt,avg_fn(resid_func(v,t,p[0],p[1],p[2],p[3])),'s',label='Average Residual')
    ax2.legend()
    
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlabel("Time/s", fontsize=16)
    ax3.set_ylabel("Averaged residual/rpm", fontsize=14)
    ax3.set_title("Average residual vs Time", fontsize=14)
    ax3.plot(tt,avg_fn(resid_func(v,t,p[0],p[1],p[2],p[3])),'g',label='Average Residual')
    p3, pc3 = optimize.curve_fit(fit_func,tt,avg_fn(resid_func(v,t,p[0],p[1],p[2],p[3])),p0=[0.6,0.03,-1.4,0])
    ax3.plot(tt, fit_func(tt,p3[0],p3[1],p3[2],p3[3]),'b',label='Average Fitted function')
    sd3 = np.sqrt(np.diag(pc3))
    r3 = stats.linregress(avg_fn(resid_func(v,t,p[0],p[1],p[2],p[3])),fit_func(tt,p3[0],p3[1],p3[2],p3[3]))[2]
    chi_sqr3 = stats.chisquare(avg_fn(resid_func(v,t,p[0],p[1],p[2],p[3])),fit_func(tt,p3[0],p3[1],p3[2],p3[3]))[0]
    ax3.text(150, 0.3, 'a = '+str(p3[0])+'±'+ str(sd3[0])
    +'\n'+'b = '+str(p3[1])+'±'+str(sd3[1])
    +'\n'+'c = '+str(p3[2])+'±'+str(sd3[2])
    +'\n'+'d = '+str(p3[3])+'±'+str(sd3[3])
    +'\n'+'r-squared = '  +str(r3**2)
    +'\n'+'chi-squared = '+str(chi_sqr3)
    )
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlabel("Time/s", fontsize=16)
    ax4.set_ylabel("Improved function/rpm", fontsize=14)
    ax4.set_title("Improved function vs Time", fontsize=14)
    ax4.plot(t,improv_func(v,t,p3[0],p3[1],p3[2],p3[3]),'r.',label='Improved plot')
    ax4.plot(t,v,'b.',label='data')
    chi_sqr4 = stats.chisquare(v,improv_func(v,t,p3[0],p3[1],p3[2],p3[3]))[0]
    r4 = stats.linregress(v,improv_func(v,t,p3[0],p3[1],p3[2],p3[3]))[2]
    ax4.text(200, 150,'r-squared = '+str(r4**2)
    +'\n'+'chi-squared = '+str(chi_sqr4)
    )
    ax4.legend()
    plt.savefig('Done.jpg')
    
main()
