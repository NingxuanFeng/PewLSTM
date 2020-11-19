import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
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
from torch.autograd import Variable
from sklearn.preprocessing import minmax_scale 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import csv

class pew_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, weather_size: int):
        super(pew_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weather_size = weather_size

        # input gate
        self.w_ix = Parameter(Tensor(hidden_size, input_size))
        self.w_ih = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ie = Parameter(Tensor(hidden_size, hidden_size))  # this vector in paper is "w_fe"
        self.b_i = Parameter(Tensor(hidden_size, 1))

        # forget gate
        self.w_fx = Parameter(Tensor(hidden_size, input_size))
        self.w_fo = Parameter(Tensor(hidden_size, hidden_size))
        self.w_fe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(hidden_size, 1))

        # output gate
        self.w_ox = Parameter(Tensor(hidden_size, input_size))
        self.w_oh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_oe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(hidden_size, 1))
        
        # cell
        self.w_gx = Parameter(Tensor(hidden_size, input_size))
        self.w_gh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(Tensor(hidden_size, 1))

        # ho
        self.w_d = Parameter(Tensor(hidden_size, input_size))
        self.w_w = Parameter(Tensor(hidden_size, input_size))
        self.w_m = Parameter(Tensor(hidden_size, input_size))
        self.w_t = Parameter(Tensor(hidden_size, hidden_size))
        self.w_e = Parameter(Tensor(hidden_size, weather_size))
        self.b_e = Parameter(Tensor(hidden_size, 1)) # this vector in paper is "b_f"

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x_input, x_weather):
        """Forward
        Args:
            inputs: [batch_size, seq_size, input_size]
            weathers: [batch_size, seq_size, weather_size]
        """

        batch_size, seq_size, input_dim = x_input.size()

        h_output = torch.zeros(batch_size, seq_size, self.hidden_size)
        c_output = torch.zeros(batch_size, seq_size, self.hidden_size)

        for b in range(batch_size):
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()

            for t in range(24):
                # day
                if b < 1:
                    h_d = torch.zeros(1, self.input_size).t()
                else:
                    h_d = x_input[b-1, t, :].unsqueeze(0).t()
                
                # week
                if b < 7:
                    h_w = torch.zeros(1, self.input_size).t()
                else:
                    h_w = x_input[b-7, t, :].unsqueeze(0).t()

                # month
                if b < 30:
                    h_m = torch.zeros(1, self.input_size).t()
                else:
                    h_m = x_input[b-30, t, :].unsqueeze(0).t()

                x = x_input[b, t, :].unsqueeze(0).t()  # [input_dim, 1]
                weather_t = x_weather[b, t, :].unsqueeze(0).t()  # [weather_dim, 1]
                
                #replace h_t with ho
                h_o = torch.sigmoid(self.w_d @ h_d + self.w_w @ h_w + self.w_t @ h_t +
                                    self.w_m @ h_m + self.w_t @ h_t)
                e_t = torch.sigmoid(self.w_e @ weather_t + self.b_e)
                # input gate
                i = torch.sigmoid(self.w_ix @ x + self.w_ih @ h_o +
                                    self.w_ie @ e_t + self.b_i)
                # cell
                g = torch.tanh(self.w_gx @ x + self.w_gh @ h_o
                                + self.b_g)
                # forget gate
                f = torch.sigmoid(self.w_fx @ x + self.w_fo @ h_o +
                                self.w_fe @ e_t + self.b_f)
                
                # output gate
                o = torch.sigmoid(self.w_ox @ x + self.w_oh @ h_t +
                                    self.w_oe @ e_t + self.b_o)

                c_next = f * c_t + i * g  # [hidden_dim, 1]
                h_next = o * torch.tanh(c_next)  # [hidden_dim, 1]

                h_output[b, t] = h_next.t().squeeze(0)
                c_output[b, t] = c_next.t().squeeze(0)

                h_t = h_next
                c_t = c_next

        return (h_output, c_output)