#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:10:56 2017

@author: prudhvi
"""

import sys
import os
#os.chdir('MLSP')
import numpy as np
import gensim
np.random.seed(2)
maxim = 32
model = gensim.models.KeyedVectors.load_word2vec_format('/home/prudhvi/MLSP/GoogleNews-vectors-negative300.bin', binary=True)  

#maxim = 36
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
#model = open('glove_100d_pickle.txt', 'r')
#model = pickle.load(model)

'''
files = open('glove.6B.300d.txt').read().splitlines()

diction = {}
import copy
print "Load Done"
count = 0
for f in files:
    vals = f.split()
    count += 1
    diction[vals[0]] = np.array(vals[1:])
    print count

model = copy.deepcopy(diction)
'''
def acquire_data(filename, model, maxim, index):
    loc_data = []
    for i in range(0, len(filename)):
        new_list = []
        current = filename[i].split()
        for c in current:
            if c in model:
                new_list.append(model[c])

        new_list = np.array(new_list)
    
        length = new_list.shape[0]
        sparser = np.zeros((maxim - length) * 300)
        new = np.reshape(new_list, (length * 300))

        vect = np.hstack((new, sparser))
        loc_data.append(vect)
        #print i

    loc_data = np.array(loc_data)
    loc_targets = [index] * len(filename)
    
    return loc_data, np.array(loc_targets)

import os

r = sys.argv[1]
files = os.listdir('sub_categories_test/' + r + '/')#['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
data = []
targets = []

for f in range(0, len(files)):
    filename = open('sub_categories/' + r + '/'+ files[f], 'r').read().splitlines()
    loc_data, loc_targets = acquire_data(filename, model, maxim, f)
    print loc_data.shape, loc_targets.shape           
    #filename.close()
    data.append(loc_data)
    targets.append(loc_targets)

data = np.array(data)
targets = np.array(targets)

data = np.vstack(data)
targets = np.hstack(targets)

x_train = data
y_train = targets


data = []
targets = []
import os
files = os.listdir('sub_categories_test/' + r + '/')#['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
for f in range(0, len(files)):
    filename = open('sub_categories_test/' + r + '/' +files[f], 'r').read().splitlines()
    loc_data, loc_targets = acquire_data(filename, model, maxim, f)
    print loc_data.shape, loc_targets.shape           
    #filename.close()
    data.append(loc_data)
    targets.append(loc_targets)

data = np.array(data)
targets = np.array(targets)

data = np.vstack(data)
targets = np.hstack(targets)

x_test = data
y_test = targets
#from sklearn.model_selection import train_test_split
print x_test.shape
#print e
#x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size= 0.2, random_state = 2)
from keras.utils import np_utils

x_train = np.reshape(x_train, (x_train.shape[0], maxim, 300, 1))
x_test = np.reshape(x_test, (x_test.shape[0], maxim, 300, 1))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

epochs = sys.argv[2]

from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, merge, Input, Dense, Dropout

inputs = Input(shape= x_train.shape[1:], dtype= 'float32')


conv_1 = Conv2D(500, 2, 300, activation= 'relu')(inputs)
conv_2 = Conv2D(500, 3, 300, activation= 'relu')(inputs)
conv_3 = Conv2D(500, 4, 300, activation= 'relu')(inputs)
conv_4 = Conv2D(500, 5, 300, activation= 'relu')(inputs)


max_pool_1 = MaxPooling2D(pool_size=(30, 1 ))(conv_1)
max_pool_2 = MaxPooling2D(pool_size=(29, 1 ))(conv_2)
max_pool_3 = MaxPooling2D(pool_size=(28, 1 ))(conv_3)
max_pool_4 = MaxPooling2D(pool_size=(27, 1))(conv_4)

merged = merge([max_pool_1, max_pool_2, max_pool_3, max_pool_4], mode= 'concat')
flatten = Flatten()(merged)
full_conn = Dense(512, activation= 'tanh')(flatten)
dropout_1 = Dropout(0.7)(full_conn)
full_conn_2 = Dense(256, activation= 'tanh')(dropout_1)
dropout_2 = Dropout(0.5)(full_conn_2)
output = Dense(len(files), activation= 'softmax')(dropout_2)

model = Model(input= inputs, output= output)

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])
model.fit(x_train, y_train, batch_size= 50, nb_epoch= int(epochs)) #validation_data =(x_test, y_test))

model.save('word2vec_' + r + '_model_' +epochs+'.h5')

#from keras.models import load_model
#model = load_model('model_sub_categ_desc_500_fil_pool_2_str_20_4_dropo_2.h5')
#model = load_model(sys.argv[2])
predictions = model.predict(x_test)
predictions = np_utils.categorical_probas_to_classes(predictions)
originals = np_utils.categorical_probas_to_classes(y_test)
lend = len(predictions) * 1.0
print lend
print np.sum(predictions == originals)
print np.sum(predictions == originals)/lend

from sklearn.metrics import confusion_matrix

print confusion_matrix(originals, predictions) #labels = ['abbr', 'desc', 'enty', 'hum', 'loc', 'num'])
