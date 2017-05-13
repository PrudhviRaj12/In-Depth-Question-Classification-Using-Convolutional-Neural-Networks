#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:12:33 2017

@author: prudhvi
"""
import sys
import pickle
import os
#os.chdir('MLSP')
import numpy as np
import gensim
np.random.seed(2)
placeholder = np.random.rand(100)
maxim = 40
import pickle
import copy

print "Loading Wiki Glove Model"

files = open('/home/prudhvi/MLSP/glove.6B.100d.txt').read().splitlines()

diction = {}

count = 0
for f in files:
    vals = f.split()
    count += 1
    diction[vals[0]] = np.array(vals[1:])
    #print count
model = copy.deepcopy(diction)
#files = open('glove_50d_pickle.txt', 'rb')
#model = pickle.load(files)
print "Load Done"
#print "Loading Word2Vec Model"
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
#print "Load Done"
#model = open('glove_100d_pickle.txt', 'r')
#model = pickle.load(model)

print "Loading Twitter Glove Model"
files = open('/home/prudhvi/MLSP/glove.twitter.27B.100d.txt').read().splitlines()

diction = {}

print "Load Done"
count = 0
for f in files:
    vals = f.split()
    count += 1
    diction[vals[0]] = np.array(vals[1:])
    #print count 

model2 = copy.deepcopy(diction)

d = 100
def acquire_data(filename, model, maxim, index):
    loc_data = []
    for i in range(0, len(filename)):
        new_list = []
        #print filename[i]
        current = filename[i].split()
        for c in current:
            if c in model:
                new_list.append(model[c])
            else:	
                new_list.append(placeholder)

        new_list = np.array(new_list)
        #print new_list.shape, new_list[0].shape  
        length = new_list.shape[0]
        #print length
        #if length <= 32:
        sparser = np.zeros((maxim - length) * d)
        new = np.reshape(new_list, (length * d))

        vect = np.hstack((new, sparser))
        loc_data.append(vect)
        #else:
	#    loc_data.append(new_list)
        #print i

    loc_data = np.array(loc_data)
    loc_targets = [index] * len(filename)
    
    return loc_data, np.array(loc_targets)

data_w2v = []
targets = []
#files = ['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
r = sys.argv[1]
epochs = sys.argv[2]
files = os.listdir('sub_categories_test/' + r + '/')
for f in range(0, len(files)):
    #filename = open('train_5000/' + files[f], 'r').read().splitlines()
    filename = open('sub_categories/' + r + '/'+ files[f], 'r').read().splitlines()
    #print filename
    loc_data, loc_targets = acquire_data(filename, model, maxim, f)
    #glove_data, glove_targets = acquire_data(filename, model2, maxim, f)
    print loc_data.shape, loc_targets.shape           
    #print glove_data.shape, glove_targets.shape
    #filename.close()
    data_w2v.append(loc_data)
    targets.append(loc_targets)

data_w2v = np.array(data_w2v)
targets = np.array(targets)

data_w2v = np.vstack(data_w2v)
targets = np.hstack(targets)

print data_w2v.shape
print targets.shape
data_glove = []
#targets = []
for f in range(0, len(files)):
    filename = open('sub_categories/' + r + '/'+ files[f], 'r').read().splitlines()
    loc_data, loc_targets = acquire_data(filename, model2, maxim, f)
    #glove_data, glove_targets = acquire_data(filename, model2, maxim, f)
    print loc_data.shape, loc_targets.shape
    #print glove_data.shape, glove_targets.shape
    #filename.close()
    data_glove.append(loc_data)
    #targets.append(loc_targets)

data_glove = np.array(data_glove)
data_glove = np.vstack(data_glove)

print '\n\n\n'
print data_w2v.shape
print data_glove.shape
#del model2
data = np.stack((data_w2v, data_glove))
data = data.reshape((data_w2v.shape[0], data_w2v.shape[1], 2))

x_train = data
y_train = targets

print x_train.shape
print y_train.shape

data_w2v = []
targets = []
#files = ['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
for f in range(0, len(files)):
    filename = open('sub_categories_test/' + r + '/'+ files[f], 'r').read().splitlines()
    loc_data, loc_targets = acquire_data(filename, model, maxim, f)
    #glove_data, glove_targets = acquire_data(filename, model2, maxim, f)
    print loc_data.shape, loc_targets.shape
    #print glove_data.shape, glove_targets.shape
    #filename.close()
    data_w2v.append(loc_data)
    targets.append(loc_targets)

data_w2v = np.array(data_w2v)
targets = np.array(targets)

data_w2v = np.vstack(data_w2v)
targets = np.hstack(targets)

print data_w2v.shape
print targets.shape

data_glove = []
#targets = []
for f in range(0, len(files)):
    filename = open('sub_categories_test/' + r + '/'+ files[f], 'r').read().splitlines()
    loc_data, loc_targets = acquire_data(filename, model2, maxim, f)
    #glove_data, glove_targets = acquire_data(filename, model2, maxim, f)
    print loc_data.shape, loc_targets.shape
    #print glove_data.shape, glove_targets.shape
    #filename.close()
    data_glove.append(loc_data)
    #targets.append(loc_targets)

data_glove = np.array(data_glove)
data_glove = np.vstack(data_glove)

print '\n\n\n'
print data_w2v.shape
print data_glove.shape
#del model2
data = np.stack((data_w2v, data_glove))
data = data.reshape((data_w2v.shape[0], data_w2v.shape[1], 2))

x_test= data
y_test=  targets 

'''
data = []
targets = []
files = ['abbr.txt', 'desc.txt', 'enty.txt', 'hum.txt', 'loc.txt', 'num.txt']
for f in range(0, len(files)):
    filename = open('test_500/' + files[f], 'r').read().splitlines()
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
from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size= 0.2, random_state = 2)
'''
from keras.utils import np_utils
x_train = np.reshape(x_train, (x_train.shape[0], maxim, d, 2, 1))
x_test = np.reshape(x_test, (x_test.shape[0], maxim, d, 2, 1))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, merge, Input, Dense, Dropout
from keras.layers import Conv3D, MaxPooling3D
inputs = Input(shape= x_train.shape[1:], dtype= 'float32')


conv_1 = Conv3D(500, 2, d, 2, activation= 'relu')(inputs)
conv_2 = Conv3D(500, 3, d, 2, activation= 'relu')(inputs)
conv_3 = Conv3D(500, 4, d, 2, activation= 'relu')(inputs)
conv_4 = Conv3D(500, 5, d, 2, activation= 'relu')(inputs)


max_pool_1 = MaxPooling3D(pool_size=(37, 1, 1 ))(conv_1)
max_pool_2 = MaxPooling3D(pool_size=(36, 1, 1 ))(conv_2)
max_pool_3 = MaxPooling3D(pool_size=(35, 1, 1 ))(conv_3)
max_pool_4 = MaxPooling3D(pool_size=(34, 1, 1 ))(conv_4)

merged = merge([max_pool_1, max_pool_2, max_pool_3, max_pool_4], mode= 'concat')
flatten = Flatten()(merged)
full_conn = Dense(512, activation= 'tanh')(flatten)
dropout_1 = Dropout(0.5)(full_conn)
full_conn_2 = Dense(256, activation= 'tanh')(dropout_1)
dropout_2 = Dropout(0.5)(full_conn_2)
output = Dense(len(files), activation= 'softmax')(dropout_2)

model = Model(input= inputs, output= output)

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])

#from keras.models import load_model
#model = load_model('multi_channel_model_test_10.h5')
model.fit(x_train, y_train, batch_size= 50, nb_epoch= int(epochs))
model.save('multi_channel_desc_model_test_' + epochs + '.h5')

#from keras.models import load_model
#model = load_model('multi_channel_model_test_14.h5')
#model = load_model(sys.argv[1])
predictions = model.predict(x_test)
predictions = np_utils.categorical_probas_to_classes(predictions)
originals = np_utils.categorical_probas_to_classes(y_test)

print np.sum(predictions == originals)
lend = len(predictions) * 1.0
print lend
print np.sum(predictions == originals)/lend

from sklearn.metrics import confusion_matrix

print confusion_matrix(originals, predictions) #labels = ['abbr', 'desc', 'enty', 'hum', 'loc', 'num'])

