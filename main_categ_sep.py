# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:09:24 2017

@author: prudh
"""

import numpy as np
import sys
import os
fi = sys.argv[1]
direc = sys.argv[2]
#os.makedirs(direc)
filename = open(fi, 'r').read().splitlines()

import re

array = []
for i in range(0, len(filename)):
    first_element = filename[i].split()[0]
    new = re.sub(':', ' ', first_element)
    new = new.split()
    array.append(new + filename[i].split()[1:])
array = np.array(array)

collect_targets = []
for i in range(0, len(array)):
    collect_targets.append(array[i][0])

collect_targets = np.array(collect_targets)
targets = np.unique(collect_targets)

#remove secondary categories
new_array = []
for a in array:
    new_array.append([a[0]] + a[2:])
    

new_array_2 = []
max_length = []
for a in array:
    new_array_2.append(a[2:])
    max_length.append(len(a[2:]))
new_array_2 = np.array(new_array_2)

new = []
for n in new_array_2:
    convert_to_string = ' '.join(n)
    new.append(re.sub('[^A-Za-z0-9]+', ' ', convert_to_string))

new = np.array(new)
#remove special characters 

#write to files
new_array = np.array(new_array)
for t in range(len(targets)):  
    current = targets[t]
    current_file = open(direc +'/' + str(current.lower()) + '.txt', 'w')
    for a in new_array:
        #print a[0] == current
        if (a[0] == current):
            new = ' '
            for c in a[1:]:
                new = new + c + ' '
            #print new
            current_file.write(new + '\n')
    current_file.close()
            #current_file.write('\n')
