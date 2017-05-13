#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 00:17:03 2017

@author: prudhvi
"""

import numpy as np
from keras.models import load_model
import gensim
import os
import sys
#os.chdir('MLSP')
import re
filename = sys.argv[1]
files = open(filename).read().splitlines()

text = []
targ1 = []
targ2 = []


for f in files:
    curr = re.sub('[^A-Za-z0-9]+', ' ', f)
    #print curr
    curr = curr.split()
    #print curr
    #first_element = curr[0]    #print first_element
    #new = re.sub(':', ' ', first_element)
    new_list = []
    text.append(curr[2:])
    #for qu in question:
    #	new_list.append(re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", qu))
    #text.append(new_list)
    #new = new.split()
    targ1.append(curr[0])
    targ2.append(curr[1])

np.random.seed(2)
maxim = 32
print "Loading Pre-trained Word2Vec Data"
model = gensim.models.KeyedVectors.load_word2vec_format('/home/prudhvi/MLSP/GoogleNews-vectors-negative300.bin', binary=True)  
print "Loading Complete "
def single_data_construct(question, model, maxim):
    print question
    new_list = []
    current = question
    #current = question.split()
    for c in current:
        if c in model:
            #print c
            new_list.append(model[c])
    
    new_list = np.array(new_list)
    
    length = new_list.shape[0]
    sparser = np.zeros((maxim - length) * 300)
    new = np.reshape(new_list, (length * 300))

    vect = np.hstack((new, sparser))
        
    return vect


#layer_1_model = load_model('500_filter_4_convs_adam_50_14_full_test_dropouts_2.h5')

layer_1_model = load_model('500_filter_4_convs_adam_50_14_full_test_dropouts_2.h5')
layer_2_model_abbr = load_model('model_sub_categ_abbr_500_fil_pool_2_str_20_10_dropo_2.h5')
layer_2_model_desc = load_model('model_sub_categ_desc_500_fil_pool_2_str_20_4_dropo_2.h5')
layer_2_model_enty = load_model('model_sub_categ_enty_500_fil_pool_2_srt_50_88_dropo_2_1_new.h5')
layer_2_model_hum = load_model('model_sub_categ_hum_500_fil_pool_2_srt_50_11_dropo_2_1_new.h5')
layer_2_model_loc = load_model('model_sub_categ_loc_500_fil_pool_2_srt_50_36_dropo_2_1_new.h5')
layer_2_model_num = load_model('model_sub_categ_num_500_fil_pool_2_srt_50_41_dropo_2_1_new.h5')

'''
layer_1_model = load_model('full_glove_model_37_iter_final.h5')
layer_2_model_abbr = load_model('glove_abbr_10_iter_model.h5')
layer_2_model_desc = load_model('glove_desc_9_iter_model.h5')
layer_2_model_enty = load_model('glove_enty_15_iter_model.h5')
layer_2_model_hum = load_model('glove_hum_4_iter_model.h5')
layer_2_model_loc = load_model('glove_loc_4_iter_model.h5')
layer_2_model_num = load_model('glove_num_9_iter_model.h5')
'''
'''
layer_1 = {0: 'Abbreviation',
           1: 'Description',
           2: 'Entity',
           3: 'Human',
           4: 'Location',
           5: 'Numeric'}
'''

layer_1 = {0: 'ABBR',
           1: 'DESC',
           2: 'ENTY',
           3: 'HUM',
           4: 'LOC',
           5: 'NUM'}


layer_2_abbr = {0: 'exp',
                1: 'abbr'}


layer_2_desc = {0: 'reason',
                    1: 'def',
                    2: 'desc',
                    3: 'manner'}

layer_2_enty = {0: 'food',
                    1: 'event',
                    2: 'lang',
                    3: 'other',
                    4: 'animal',
                    5: 'product',
                    6: 'techmeth',
                    7: 'substance',
                    8: 'currency',
                    9: 'body',
                    10: 'dismed',
                    11: 'veh',
                    12: 'plant',
                    13: 'sport',
                    14: 'color',
                    15: 'instru',
                    16: 'termeq'}

layer_2_hum = {0: 'title',
                   1: 'gr',
                   2: 'ind',
                   3: 'desc'}

layer_2_loc = {0: 'other',
                   1: 'state',
                   2: 'city',
                   3: 'country',
                   4: 'mount'}

layer_2_num = {0: 'weight',
                   1: 'other',
                   2: 'ord',
                   3: 'temp',
                   4: 'dist',
                   5: 'money',
                   6: 'date',
                   7: 'code',
                   8: 'period',
                   9: 'count',
                   10: 'perc',
                   11: 'speed',
                   12: 'volsize'}

pred_main = []
pred_sec = []
text = np.array(text)
for t in range(0, len(text)):
    current = single_data_construct(np.array(text[t]), model, maxim)
    current = np.reshape(current, (1, 32, 300, 1))

    predicted = layer_1_model.predict(current)

    argument = np.argmax(predicted)

    print "Predicted: " + str(layer_1[argument]) + " Confidence : " + str(round(predicted[0][argument] * 100, 3)) + "%"
    predicted = layer_1[argument]
    pred_main.append(predicted)
    if predicted == layer_1[0]:

        predicted_abbr = layer_2_model_abbr.predict(current)
        
        argument_abbr = np.argmax(predicted_abbr)

        print "Predicted Abbreviation Sub Categ: " + str(layer_2_abbr[argument_abbr]) + " Confidence : " + str(round(predicted_abbr[0][argument_abbr] * 100, 3)) + "%"
        pred_sec.append(layer_2_abbr[argument_abbr])
    if predicted == layer_1[1]:

        predicted_desc = layer_2_model_desc.predict(current)
    
        argument_desc = np.argmax(predicted_desc)

        print "Predicted Description Sub Categ: " +str(layer_2_desc[argument_desc]) + " Confidence : " + str(round(predicted_desc[0][argument_desc] * 100, 3)) + "%"
        pred_sec.append(layer_2_desc[argument_desc])
    if predicted == layer_1[2]:

        predicted_enty = layer_2_model_enty.predict(current)

        argument_enty = np.argmax(predicted_enty)

        print "Predicted Entity Sub Categ: " +str(layer_2_enty[argument_enty]) + " Confidence : " + str(round(predicted_enty[0][argument_enty] * 100, 3)) + "%"
        pred_sec.append(layer_2_enty[argument_enty])
    if predicted == layer_1[3]:

        predicted_hum = layer_2_model_hum.predict(current)
    
        argument_hum = np.argmax(predicted_hum)

        print "Predicted Human Sub Categ: " +str(layer_2_hum[argument_hum]) + " Confidence : " + str(round(predicted_hum[0][argument_hum] * 100, 3)) + "%"
        pred_sec.append(layer_2_hum[argument_hum])
    if predicted == layer_1[4]:

        predicted_loc = layer_2_model_loc.predict(current)
    
        argument_loc = np.argmax(predicted_loc)

        print "Predicted Location Sub Categ: " +str(layer_2_loc[argument_loc])+ " Confidence : " + str(round(predicted_loc[0][argument_loc] * 100, 3)) + "%"
        pred_sec.append(layer_2_loc[argument_loc])
    if predicted == layer_1[5]:

        predicted_num = layer_2_model_num.predict(current)
    
        argument_num = np.argmax(predicted_num)

        print "Predicted Numeric Sub Categ: " +str(layer_2_num[argument_num]) + " Confidence : " + str(round(predicted_num[0][argument_num] * 100, 3)) + "%"
        pred_sec.append(layer_2_num[argument_num])
targets = np.array(targ1)
predictions = np.array(pred_main)


print len(targets)
print len(predictions)
print np.sum(targets == predictions)

sec_targets = np.array(targ2)
predictions_2 = np.array(pred_sec)
print np.sum(sec_targets == predictions_2)

#for i in range(0, len(targets)):
    #print targets[i], predictions[i]
#print sec_targets
#print predictions_2
from sklearn.metrics import confusion_matrix

print confusion_matrix(targets, predictions)
