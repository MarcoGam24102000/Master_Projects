# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 08:30:17 2022

@author: marco
"""

import matplotlib.pyplot as plt
import sys
import csv
import time
import os
import numpy

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path: 
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: 
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

if len(sys.argv) == 5:
    vbr_indic = sys.argv[2]
    bufferData = int(sys.argv[3])
    filenameCSV_dir = sys.argv[4]
    
dir_splitted = splitall(filenameCSV_dir)

dirBuilt = "C:/" 

for pos in range(1,len(dir_splitted)):
    dirBuilt = dirBuilt + str(dir_splitted[pos]) + "/"

dirBuilt += "buffer_management_vbr.csv"

timeList = []
startPointBuffer = []
endPointBuffer = []
 
with open(dirBuilt, 'r') as file:
    reader = csv.reader(file) 
    for rowNumber, row in enumerate(reader):    
        if rowNumber > 0:           
            timeList.append(float(row[0]))
            startPointBuffer.append(round(float(row[1])))
            endPointBuffer.append(round(float(row[2])))
           
size_dataset = len(timeList)

plt.figure()
plt.grid()
plt.title('VBR - Buffer management along time') 
plt.plot(timeList, startPointBuffer)
plt.plot(timeList, endPointBuffer)
plt.xlabel('Time (s)')
plt.ylabel('Buffer data length (bytes)')
plt.legend(['Input Buffer', 'Output Buffer']) 
plt.savefig('VBR_graph.png')
plt.show()      
