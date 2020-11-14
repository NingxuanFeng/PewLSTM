#!/usr/bin/env python
# coding: utf-8

import xlrd
import numpy as np
from math import sqrt
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
import random as rd
import calendar
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import minmax_scale 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import csv


# Global variables 
record_path = '../data/record/'
weather_path = '../data/weather/'

park_all_cnt = 10
weather_all_cnt = 6

park_table_id = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']
park_weather_idx = [0,0,1,1,1,2,2,2,2,2]
weather_name = ['Ningbo','Ningbo Yinzhou','Changsha']


# util function

def read_park_table(index, debug = False):
    park_table_path = record_path + park_table_id[index] + '.csv'
    park_book = pd.read_csv(park_table_path,encoding='ISO-8859-1')##########
    if debug:
        print('open table ' + park_table_name[i] + ' with lines ' + str(len(park_book)))
    return park_book

def read_weather_table(index, debug = False):
    weather_table_path = weather_path + str(index) + '.csv'
    weather_book = pd.read_csv(weather_table_path,encoding='ISO-8859-1')
    if debug:
        print ('open table ' + weather_name[i] + ' with lines ' + str(len(weather_book)))
    return weather_book

def trans_record_to_count(data, debug = False):
    invalid_record = 0
    valid_record = 0
    p_dict = {}
    for stime,etime in zip(data['Lockdown Time'],data['Lockup Time']):
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")##########
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")#########
        # Converts start and end times to seconds
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        # A parking record which has duration less than 5 mins are regard as invalid record
        if end_tsp - start_tsp <= 5*60:
            invalid_record = invalid_record + 1
            continue
        valid_record = valid_record + 1
        start_hour = int(start_tsp//(60*60))
        end_hour = int(end_tsp//(60*60))
        # Calculate the parking numbers per hour
        for j in range(start_hour,end_hour+1):
            if j not in p_dict:
                p_dict[j] = {}
                p_dict[j]['cnt'] = 1
            else:
                p_dict[j]['cnt'] = p_dict[j]['cnt'] + 1
    if debug:
        print('valid record is ' + str(valid_record))
        print('invalid record is ' + str(invalid_record))
    return p_dict

def calc_park_cnt_from_dict(p_dict, debug = False):
    if debug:
        print('calcing parking count from dict ...')
    park_cnt = []
    st = min(p_dict.keys())
    ed = max(p_dict.keys())
    for i in range(st,ed+1):
        if i in p_dict:
            park_cnt.append(p_dict[i]['cnt'])
        else:
            park_cnt.append(0)
    return park_cnt

def process_weather(data, debug= False):
    output = []
    start_h = data['DAY'][0]
    start_h = int(time.mktime(time.strptime(start_h,"%Y/%m/%d %H:%M")) // (60*60))############
    output.append(start_h)
    
    for i in range(5):
        output.append([])
    output.append({})
    for i in range(len(data['HOUR'])):
        output[1].append(data['TEM'][i])
        output[2].append(data['RHU'][i])
        output[3].append(data['WIN_S'][i])
        output[4].append(data['PRE_1h'][i])
        output[5].append(time.strptime(data['DAY'][i],"%Y/%m/%d %H:%M").tm_wday)##############
        output[6][int(time.mktime(time.strptime(data['DAY'][i],"%Y/%m/%d %H:%M")) // (60*60))] = i############
    return output
def invalid(w_list,idx):
    if w_list[1][idx] > 999:
        return True
    if w_list[2][idx] > 999:
        return True
    if w_list[3][idx] > 999:
        return True
    if w_list[4][idx] > 999:
        return True
    return False
def gen_series(park_cnt, weather_rec, start_h, end_h, debug=False):
    tt = []
    for i in range(len(park_cnt)):
        tt.append(start_h + i)
    """if debug:
        print(tt[-1])"""
    temp = []
    for i in range(5):
        temp.append([])
    for i in range(len(park_cnt)):
        if tt[i] in weather_rec[6]:
            idx = weather_rec[6][tt[i]]
            if invalid(weather_rec,idx):
                continue
            temp[0].append(park_cnt[i])
            temp[1].append(weather_rec[1][idx])
            temp[2].append(weather_rec[2][idx])
            temp[3].append(weather_rec[3][idx])
            temp[4].append(weather_rec[4][idx])
    #if debug:
        #print('The length of temp array is ' + str(len(temp[0])))
    
    park_cnt = pd.Series(temp[0], name='cnt')
    tem = pd.Series(temp[1], name='tem')
    rhu = pd.Series(temp[2], name='rhu')
    winds = pd.Series(temp[3], name='wind_s')
    pre_1h = pd.Series(temp[4],name='pre_ih')
    output = pd.concat([tem,rhu,winds,pre_1h,park_cnt], axis=1)
    return output

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def GetCntData(index):
    park_book = read_park_table(index)
    #weather_book = read_weather_table(park_weather_idx[index])
    p_dic = trans_record_to_count(park_book)
    park_cnt = calc_park_cnt_from_dict(p_dic)
    park_cnt = minmax_scale(park_cnt)
    data_x = park_cnt[:-1]
    data_y = park_cnt[1:]
    return (np.array(data_x),np.array(data_y))
def GetAllData(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    p_dic = trans_record_to_count(park_book)
    start_h = min(p_dic.keys())
    end_h = max(p_dic.keys())
    park_cnt = calc_park_cnt_from_dict(p_dic)
    #print(park_cnt)
    weather_rec = process_weather(weather_book)
    p_series = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    #print(p_series)
    p_series = p_series.dropna(axis=0)
    p_series = p_series.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    sclaed = scaler.fit_transform(p_series)
    reframed = series_to_supervised(sclaed, 1, 1)
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)
    return (reframed.values[:,:-1],reframed.values[:,-1])
  



def mypreprocessing(precord, weather):
    valid_record = 0
    record = []
    for stime,etime in zip(precord['Lockdown Time'],precord['Lockup Time']):
        # Parses the time string into a start time tuple and end time tuple according to the specified format
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")#######
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")#######
        # Converts start and end times to seconds
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        # A parking record which has duration less than 5 mins are regard as invalid record
        if end_tsp - start_tsp <= 5*60:
            #invalid_record = invalid_record + 1
            continue
        valid_record = valid_record + 1
        start_hour = int(start_tsp//(60*60))
        duration = int((end_tsp - start_tsp)//60) # minute
        record.append((start_hour, duration)) ###
    #print(record)
    w_dict = {}
    for i in range(len(weather['HOUR'])):
        s_hour = int(time.mktime(time.strptime(weather['DAY'][i],"%Y/%m/%d %H:%M"))//(60*60))#######
        """if i == 0:
            print(s_hour)
        elif i == len(weather['HOUR'])-1:
            print(s_hour)"""
        
        if s_hour not in w_dict:
            w_dict[s_hour] = {}
            w_dict[s_hour]['RHU'] = weather['RHU'][i]
            w_dict[s_hour]['TEM'] = weather['TEM'][i]
            w_dict[s_hour]['WIN_S'] = weather['WIN_S'][i]
            w_dict[s_hour]['wday'] = time.strptime(weather['DAY'][i],"%Y/%m/%d %H:%M").tm_wday########
            w_dict[s_hour]['hour'] = time.strptime(weather['DAY'][i],"%Y/%m/%d %H:%M").tm_hour#######
    data = []
    idx = 0
    for (s_hour, duration) in record:
        #tmp.append(,w_dict[s_hour]['TEM'],w_dict[s_hour]['WIN_S'],w_dict[s_hour]['wday'],w_dict[s_hour]['hour'],duration)
        #print(s_hour)
        if s_hour not in w_dict:
            #print(s_hour)
            continue
        data.append([])
        data[idx].append(s_hour)
        data[idx].append(w_dict[s_hour]['RHU'])
        data[idx].append(w_dict[s_hour]['TEM'])
        data[idx].append(w_dict[s_hour]['WIN_S'])
        data[idx].append(w_dict[s_hour]['wday'])
        data[idx].append(w_dict[s_hour]['hour'])
        data[idx].append(duration)
        idx+=1
    #print(data)   
    dt = []
    i = 0
    ct = 0
    while i < idx:
        j = i + 1
        avg = data[i][6]#duration
        while j < idx:
            if data[j][0] == data[i][0]:
                avg += data[j][6]
                #print(data[j])
                j = j+1
            else:
                break
        avg /= (j-i)
        dt.append([])
        for t in range(5):
            dt[ct].append(data[i][t+1])
        dt[ct].append(avg)
        i = j
        ct+=1
    return dt
def getdata(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    data = mypreprocessing(park_book, weather_book)
    return data


# In[29]:


#departure 
def ptrans_record_to_count(data, debug = False):
    invalid_record = 0
    valid_record = 0
    p_dict = {}
    for stime,etime in zip(data['Lockdown Time'],data['Lockup Time']):
        # Parses the time string into a start time tuple and end time tuple according to the specified format
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")########
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")#####
        # Converts start and end times to seconds
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        # A parking record which has duration less than 5 mins are regard as invalid record
        if end_tsp - start_tsp <= 5*60:
            invalid_record = invalid_record + 1
            continue
        valid_record = valid_record + 1
        #start_hour = int(start_tsp//(60*60))
        end_hour = int(end_tsp//(60*60))
        # Calculate the parking numbers per hour
        if end_hour not in p_dict:
            p_dict[end_hour] = {}
            p_dict[end_hour]['cnt'] = 1
        else:
            p_dict[end_hour]['cnt'] += 1
    if debug:
        print('valid record is ' + str(valid_record))
        print('invalid record is ' + str(invalid_record))
    return p_dict
#arrive 
def p2trans_record_to_count(data, debug = False):
    invalid_record = 0
    valid_record = 0
    p_dict = {}
    for stime,etime in zip(data['Lockdown Time'],data['Lockup Time']):
        # Parses the time string into a start time tuple and end time tuple according to the specified format
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")##########
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")######################
        # Converts start and end times to seconds
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        # A parking record which has duration less than 5 mins are regard as invalid record
        if end_tsp - start_tsp <= 5*60:
            invalid_record = invalid_record + 1
            continue
        valid_record = valid_record + 1
        start_hour = int(start_tsp//(60*60))
        #end_hour = int(end_tsp//(60*60))
        # Calculate the parking numbers per hour
        if start_hour not in p_dict:
            p_dict[start_hour] = {}
            p_dict[start_hour]['cnt'] = 1
        else:
            p_dict[start_hour]['cnt'] += 1
    if debug:
        print('valid record is ' + str(valid_record))
        print('invalid record is ' + str(invalid_record))
    return p_dict
def pGetCntData(index):
    park_book = read_park_table(index)
    #weather_book = read_weather_table(park_weather_idx[index])
    p_dic = ptrans_record_to_count(park_book)
    park_cnt = calc_park_cnt_from_dict(p_dic)
    park_cnt = minmax_scale(park_cnt)
    data_x = park_cnt[:-1]
    data_y = park_cnt[1:]
    return (np.array(data_x),np.array(data_y))
def p2GetCntData(index):
    park_book = read_park_table(index)
    #weather_book = read_weather_table(park_weather_idx[index])
    p_dic = p2trans_record_to_count(park_book)
    park_cnt = calc_park_cnt_from_dict(p_dic)
    park_cnt = minmax_scale(park_cnt)
    data_x = park_cnt[:-1]
    data_y = park_cnt[1:]
    return (np.array(data_x),np.array(data_y))


def pGetAllData(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    p_dic = ptrans_record_to_count(park_book)
    start_h = min(p_dic.keys())
    end_h = max(p_dic.keys())
    park_cnt = calc_park_cnt_from_dict(p_dic)
    #print(park_cnt)
    weather_rec = process_weather(weather_book)
    p_series = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    p_series = p_series.dropna(axis=0)
    p_series = p_series.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    sclaed = scaler.fit_transform(p_series)
    reframed = series_to_supervised(sclaed, 1, 1)
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)
    #print(reframed.values[:,:-1])
    return (reframed.values[:,:-1],reframed.values[:,-1])
def p2GetAllData(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    p_dic = p2trans_record_to_count(park_book)
    start_h = min(p_dic.keys())
    end_h = max(p_dic.keys())
    park_cnt = calc_park_cnt_from_dict(p_dic)
    #print(park_cnt)
    weather_rec = process_weather(weather_book)
    p_series = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    #print(p_series)
    p_series = p_series.dropna(axis=0)
    p_series = p_series.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    sclaed = scaler.fit_transform(p_series)
    reframed = series_to_supervised(sclaed, 1, 1)
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)
    #print(reframed.values[:,:-1])
    return (reframed.values[:,:-1],reframed.values[:,-1])

class LAll_LSTM(nn.Module):
    # timemode: 0 for day, 1 for week, 2 for 
    def __init__(self):
        super(LAll_LSTM, self).__init__()
        self.pa1 = 64
        self.pa2 = 64
        self.lstm1 = nn.LSTMCell(5, self.pa1)#LSTMCell(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.pa1, self.pa2)
        self.layer = nn.Linear(self.pa2,1)
        
    def forward(self, input, future = 0):
        outputs = []
        
        h_t = Variable(
            torch.zeros(input.size(0), self.pa1), requires_grad=False)
        c_t = Variable(
            torch.zeros(input.size(0), self.pa1), requires_grad=False)
        h_t2 = Variable(
            torch.zeros(input.size(0), self.pa2), requires_grad=False)
        c_t2 = Variable(
            torch.zeros(input.size(0), self.pa2), requires_grad=False)
        h_t_array = []
        c_t_array = []
        h_t2_array = []
        idx = 0
        threshold = [0.125,0.05,0.025]
        
        for i, input_t in enumerate(input.chunk(1, dim=1)):#####
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            ts = 0
            outp = 0
            if idx >= 24:
                outp += threshold[-2]*outputs[idx-24]
                ts += threshold[-2]
            if idx >= 24*7:
                outp += threshold[-3]*outputs[idx-24*7]
                ts += threshold[-3]
            if idx >= 24*30:
                outp += threshold[-1]*outputs[idx-30*24]
                ts += threshold[-1]
            outp += c_t2*(1-ts)
            outp = self.layer(outp)
            outputs += [outp]
            idx += 1

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
def LAll_LSTM_Predict(x,y):
    l = (int)(0.75*len(x))
    #print(l)
    train_x = x[:l]
    train_y = y[:l]
    test_x = x[l:]
    test_y = y[l:]
    
    train_x = train_x.reshape(-1,5)
    train_y = train_y.reshape(-1,1)
    
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    
    model = LAll_LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # epoch = 500
    for i in range(300):
        #print(i)
        out = model(train_x)
        loss = loss_function(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
 
        #if (i+1) % 10 == 0:
           # print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))
    model = model.eval()
    test_x = test_x.reshape(-1,5)
    test_y = test_y.reshape(-1,1)
    
    test_x = torch.from_numpy(test_x)
    pred_test = model(test_x)
    pred_test = pred_test.view(-1).data.numpy()
    pred_test = np.concatenate((np.zeros(1), pred_test))
    #print(len(pred_test),len(test_y),len(test_x))
    #for i in range(len(pred_test)):
        #print(pred_test[i], test_y[i])

    pred_test = pred_test[1:]
    p = 0.0
    k = 0
    test_y = test_y.reshape(-1,1)
    pred_test = pred_test.reshape(-1,1)
    for i in range(len(test_y)-1):
        k+=1
        if test_y[i] != 0:
            p = p + (abs(test_y[i]-pred_test[i+1])/test_y[i])
        else:
            p = p + abs(test_y[i]-pred_test[i+1])
    print('accuracy: ' + str(1-p/k))
    accuracy = 1-p/k
    return (pred_test,accuracy)

print("lpew departure 1hs:")
x, y = pGetAllData(0)
x = x.astype('float32') 
y = y.astype('float32') 
ai = LAll_LSTM_Predict(x,y)

#####all arrive 3h
print("lpew arrive 1hs:")
x, y = p2GetAllData(0)
x = x.astype('float32') 
y = y.astype('float32') 
ai = LAll_LSTM_Predict(x,y)