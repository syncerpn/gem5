# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:56:22 2023

@author: Nghia
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import math

### USER SPECIFICATION ###
NUM_CPUS = 8
NUM_L2 = 4
NUM_DIRS = 4

MESH_COL = 4
MESH_ROW = 4

config = [0,4,8,12,3,7,11,15,5,9,6,10,1,13,2,14]
### USER SPECIFICATION ###

DEVICE_LABELS = ['cpu_'+str(i) for i in range(NUM_CPUS)] + ['l2_'+str(i) for i in range(NUM_L2)] + ['dir_'+str(i) for i in range(NUM_DIRS)]
DEVICE_COLORS = ['deepskyblue'] * NUM_CPUS + ['limegreen'] * NUM_L2 + ['orangered'] * NUM_DIRS

def visualize_config(p, title=None):
    plt.subplots()
    
    save_file_name = '_'.join([str(pi) for pi in p]) + '.svg'
    for d, l, c in zip(p, DEVICE_LABELS, DEVICE_COLORS):
        x = d  % MESH_COL
        y = d // MESH_COL
        plt.scatter(x, y, c=c)
        plt.annotate(l, (x+0.1,y+0.1),fontsize=8)
    
    plt.xlim((-1,MESH_COL))
    plt.ylim((-1,MESH_ROW))
    plt.axis('off')
    
    for yi in range(MESH_ROW):
        plt.plot([0,MESH_COL-1], [yi, yi], c='gray', alpha=0.5)
        
    for xi in range(MESH_COL):
        plt.plot([xi,xi], [0, MESH_ROW-1], c='gray', alpha=0.5)
    
    if title is not None:
        plt.title(title)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_file_name)

visualize_config(config)