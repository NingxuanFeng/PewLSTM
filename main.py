import xlrd
import numpy as np
from math import sqrt
import pandas as pd
import time
import datetime
import math
import random as rd
import calendar
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from sklearn.preprocessing import minmax_scale 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import csv
from PewLSTM import pew_LSTM

HIDDEN_DIM = 1
SEQ_SIZE = 24
message = "pewLSTM"

# Global variables 
record_path = './data/record/'
weather_path = './data/weather/'

park_all_cnt = 10
weather_all_cnt = 6

park_table_id = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13']
park_weather_idx = [0,0,1,1,1,2,2,2,2,2,3,4,5]
weather_name = ['Ningbo','Ningbo Yinzhou','Changsha']

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
        output[6][int(time.mktime(time.strptime(data['DAY'][i],"%Y/%m/%d %H:%M")) // (60*60))] = i########
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

def is_valid(w_list,idx):
    flag = [1,1,1,1]  # "0" represents that this data is invalid
    for i in range(1,5):
        if w_list[i][idx] > 999:
            flag[i-1] = 0
    return flag

def valid_weather(w_list,idx): 
    flag = is_valid(w_list,idx)
    temp = [0,0,0,0]
    d = 0
    for i in range(1,5):
        if flag[i-1] == 0:
            d = idx - 1
            while (is_valid(w_list,d)[i-1] == 0):
                d -= 1
            upvalue = w_list[i][d]
            d = idx + 1
            while (is_valid(w_list,d)[i-1] == 0):
                d += 1
            downvalue = w_list[i][d]
            temp[i-1] = 0.5 * (upvalue + downvalue)
        else:
            temp[i-1] = w_list[i][d]
    return temp

def gen_series(park_cnt, weather_rec, start_h, end_h, debug=False):
    tt = []
    for i in range(len(park_cnt)):
        tt.append(start_h + i)
    temp = []
    for i in range(5):
        temp.append([])
    for i in range(len(park_cnt)):
        if tt[i] in weather_rec[6]:
            idx = weather_rec[6][tt[i]]
            temp[0].append(park_cnt[i])
            if invalid(weather_rec,idx):
                vld = valid_weather(weather_rec,idx)
                temp[1].append(vld[0])
                temp[2].append(vld[1])
                temp[3].append(vld[2])
                temp[4].append(vld[3])
            else:
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
    # print("park_cnt: "+str(len(park_cnt)))
    # print("weather_rec: "+str(len(weather_rec[1])))
    # print("output: "+str(len(output)))
    return output

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
#depature
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
    #park_cnt = minmax_scale(park_cnt)
    scaler = MinMaxScaler()
    park_cnt = np.array(park_cnt).reshape(-1,1)
    park_cnt = scaler.fit_transform(park_cnt)
    park_cnt = park_cnt.reshape(-1)
    data_x = park_cnt[:-1]
    data_y = park_cnt[1:]
    return (data_x,data_y,scaler)

def p2GetCntData(index):
    park_book = read_park_table(index)
    #weather_book = read_weather_table(park_weather_idx[index])
    p_dic = p2trans_record_to_count(park_book)
    park_cnt = calc_park_cnt_from_dict(p_dic)
    #park_cnt = minmax_scale(park_cnt)
    scaler = MinMaxScaler()
    park_cnt = np.array(park_cnt).reshape(-1,1)
    park_cnt = scaler.fit_transform(park_cnt)
    park_cnt = park_cnt.reshape(-1)
    data_x = park_cnt[:-1]
    data_y = park_cnt[1:]
    return (data_x,data_y,scaler)

def pGetAllData(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    p_dic = ptrans_record_to_count(park_book)
    start_h = min(p_dic.keys())
    end_h = max(p_dic.keys())
    park_cnt = calc_park_cnt_from_dict(p_dic)
    weather_rec = process_weather(weather_book)
    #p_series1 = gen_series_old(park_cnt, weather_rec, start_h, end_h,debug=True)
    p_series = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    p = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    p.fillna(value = 0,inplace=True)
    values = [0,0,0,0]
    for k in range(len(p_series)):
        values[0] += p['tem'][k]
        values[1] += p['rhu'][k]
        values[2] += p['wind_s'][k]
        values[3] += p['pre_ih'][k]
    for i in range(4):
        values[i] /= (len(p_series))
    p_series['tem'].fillna(value=values[0],inplace=True)
    p_series['rhu'].fillna(value=values[1],inplace=True)
    p_series['wind_s'].fillna(value=values[2],inplace=True)
    p_series['pre_ih'].fillna(value=values[3],inplace=True)

    p_series = p_series.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    sclaed = scaler.fit_transform(p_series)
    reframed = series_to_supervised(sclaed, 1, 1)
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)
    #print(reframed.values[:,:-1])

    s = MinMaxScaler(feature_range=(0,1))
    m = p_series
    m = series_to_supervised(m, 1, 1)
    m.drop(m.columns[[5,6,7,8]], axis=1, inplace=True)
    m1 = m.values[:,:-1]
    m2 = m.values[:,-1]
    m2 = m2.reshape(-1,1)
    # print(m2.shape)
    m22 = s.fit_transform(m2)
    return (reframed.values[:,:-1],reframed.values[:,-1],s)

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

    s = MinMaxScaler(feature_range=(0,1))
    m = p_series
    m = series_to_supervised(m, 1, 1)
    m.drop(m.columns[[5,6,7,8]], axis=1, inplace=True)
    m1 = m.values[:,:-1]
    m2 = m.values[:,-1]
    m2 = m2.reshape(-1,1)
    m22 = s.fit_transform(m2)
    return (reframed.values[:,:-1],reframed.values[:,-1],s)


class Pew_LSTM(nn.Module):
    # timemode: 0 for day, 1 for week, 2 for 
    def __init__(self):
        super(Pew_LSTM, self).__init__()
        self.lstm1 = pew_LSTM(1, HIDDEN_DIM, 4)
        self.lstm2 = pew_LSTM(HIDDEN_DIM, HIDDEN_DIM, 4)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input):  # [batch_size, seq_size, weather_size + input_dim]
        x_weather = input[:, :, :-1]  # [batch_size, seq_size, weather_size]
        x_input = input[:, :, -1].unsqueeze(2)  # [batch_size, seq_size, input_dim]

        h1, c1 = self.lstm1(x_input, x_weather)  # ([batch_size, seq_size, hidden_size], [batch_size, seq_size, hidden_size])
        h2, c2 = self.lstm2(h1, x_weather)  # ([batch_size, seq_size, hidden_size], [batch_size, seq_size, hidden_size])
        out = h2.contiguous().view(-1, HIDDEN_DIM)  # out size: (24 * batch_size, hidden_dim)
        out = self.fc(out).view(-1)  # (24 * batch_size)
        return out

def Pew_LSTM_Predict(x,y,s,pattern):
    x = torch.from_numpy(x)  # [hour_size, 5]
    y = torch.from_numpy(y)  # [hour_size]

    x = x[:((x.size(0) // 24) * 24)].reshape((x.size(0) // 24, 24, 5))
    y = y[:((y.size(0) // 24) * 24)]

    l = (int)(0.75*len(x))
    train_x = x[:l]
    train_y = y[:l*24]
    test_x = x[l:]
    test_y = y[l*24:]

    model = Pew_LSTM()
    loss_function = nn.MSELoss()

    if pattern == 0:
        model.load_state_dict(torch.load("model_P1_1h.pth"))
        model = model.eval()
        pred_test = model(test_x).cpu().detach().numpy().reshape(-1,1)
        p = 0.0
        r = 0
        k = 0
        test_y_numpy = test_y.reshape(-1,1).cpu().numpy()
        atest_y = s.inverse_transform(test_y_numpy)
        apred_test = s.inverse_transform(pred_test)
        for i in range(len(test_y_numpy)-1):
            k+=1
            if test_y_numpy[i] != 0:
                p = p + (abs(test_y_numpy[i]-pred_test[i+1])/test_y_numpy[i])
            else:
                p = p + abs(test_y_numpy[i]-pred_test[i+1])
            r += (atest_y[i]-apred_test[i+1])**2
        accuracy = (1-p/k)*100
        rmse = sqrt(r/k)
        print('accuracy: ' + str(round(accuracy[0],2))+"%"+' rmse: '+str(round(rmse,2)))
    else:
        # epoch = 500
        for i in range(500):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            out = model(train_x)
            loss = loss_function(out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))
        model = model.eval()
        pred_test = model(test_x).cpu().detach().numpy().reshape(-1,1)
        p = 0.0
        r = 0
        k = 0
        test_y_numpy = test_y.reshape(-1,1).cpu().numpy()
        atest_y = s.inverse_transform(test_y_numpy)
        apred_test = s.inverse_transform(pred_test)
        for i in range(len(test_y_numpy)-1):
            k+=1
            if test_y_numpy[i] != 0:
                p = p + (abs(test_y_numpy[i]-pred_test[i+1])/test_y_numpy[i])
            else:
                p = p + abs(test_y_numpy[i]-pred_test[i+1])
            r += (atest_y[i]-apred_test[i+1])**2
        accuracy = (1-p/k)*100
        rmse = sqrt(r/k)
        print('accuracy: ' + str(round(accuracy[0],2))+"%"+' rmse: '+str(round(rmse,2)))
    return (pred_test,accuracy)

k = 0
print("Example: P"+str(k+1))

print("pew depature 1h:")
pattern = 0
x, y, s= pGetAllData(k)
x = x.astype('float32') 
y = y.astype('float32') 
ai = Pew_LSTM_Predict(x,y,s,pattern)
