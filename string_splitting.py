# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:32:16 2018

@author: Charlotte
"""

#o = []
#while tp:
#    o.append(tp[:20])
#    tpp = tp[20:]


def split_every(n, s):
    return [ s[i:i+n] for i xrange(0, len(s), n) ]

print(split_every(20, tp))



#def split_by_n(seq, n):
#    while seq:
#        yield seq[:n]
#        seq = seq[n:]
#print(list(split_by_n(tp, 20)))



#for i in range(1,20):
   #string = tp.len(20)
   
tpp = [tp[i:i+20] for i in range(0, 0, 20)]
print(tpp)