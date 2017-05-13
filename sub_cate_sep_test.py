#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:57:29 2017

@author: prudhvi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:09:24 2017

@author: prudh
"""

import numpy as np
import os
import sys
#os.chdir('MLSP')
#filename = open('test_data.txt', 'r').read().splitlines()
fi = sys.argv[1]
#main_dir_1 = sys.argv[2]
categ=  sys.argv[2]
path = sys.argv[3]
filename = open(fi, 'r').read().splitlines()

import re

array = []
for i in range(0, len(filename)):
    first_element = filename[i].split()[0]
    #print first_element
    new = re.sub(':', ' ', first_element)
    new = new.split()
    #print new
    if new[0] == categ.upper():
        array.append([new[1]] + filename[i].split()[1:])

array = np.array(array)

collect_targets = []
for i in range(0, len(array)):
    collect_targets.append(array[i][0])

collect_targets = np.array(collect_targets)
targets = np.unique(collect_targets)

#remove secondary categories

'''
new_array = []
for a in array:
    new_array.append(a[1:])
 

new_array_2 = []
max_length = []
for a in array:
    new_array_2.append(a[1:])
    max_length.append(len(a[1:]))
new_array_2 = np.array(new_array_2)
'''
new_array_2 = array
new = []
for n in new_array_2:
    convert_to_string = ' '.join(n)
    new.append(re.sub('[^A-Za-z0-9]+', ' ', convert_to_string))

new = np.array(new)
#remove special characters 

#write to files
os.makedirs(path + categ)

new_array = np.array(new)
for t in range(len(targets)):  
    current = targets[t]
    #print current
    #os.makedirs('sub_categories/' + categ)
    current_file = open(path +str(categ) + '/'  +str(current.lower()) + '.txt', 'w')
    for a in new_array:
        #print a
        #print a.split()[0]
        #print a[0] == current
        if (a.split()[0] == current):
            new = ' '
            for c in a.split()[1:]:
                new = new + c + ' '
            #print new
            current_file.write(new + '\n')
    current_file.close()
            #current_file.write('\n')
