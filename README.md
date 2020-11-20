# PewLSTM
This is the implementation of the model and real-world parking data described in:

Feng Zhang, Ningxuan Feng, Yani Liu, Cheng Yang, Jidong Zhai, Shuhao Zhang, Bingsheng He, Jiazao Lin, Xiaoyong Du, "PewLSTM: Periodic LSTM with Weather-Aware Gating Mechanism for Parking Behavior Prediction"

Appeared in IJCAI'2020

This is a Pytorch implementation of the PewLSTM model described in paper. PewLSTM is a novel periodic weather-aware LSTM model that successfully predicts the parking behavior based on historical records, weather, environments, and weekdays.

## Abstract

In big cities, there are plenty of parking spaces, but we often ﬁnd nowhere to park. For example, New York has 1.4 million cars and 4.4 million on-street parking spaces,but it is still not easy to ﬁnd a parking place near our destination, especially during peak hours. The reason is the lack of prediction of parking behavior. If we could provide parking behavior in advance, we can ease this parking problem that affects human well-being. We observe that parking lots have periodic parking patterns, which is an important factor for parking behavior prediction. Unfortunately, existing work ignores such periodic parking patterns in parking behavior prediction, and thus incurs low accuracy. To solve this problem, we propose PewLSTM, a novel periodic weather-aware LSTM model that successfully predicts the parking behavior based on historical records, weather, environments, and weekdays. PewLSTM has been successfully integrated into a real parking space reservation system, ThsParking, which is one of the top smart parking platforms in China. Based on 452,480 real parking records in 683 days from 10 parking lots, PewLSTM yields 85.3% parking prediction accuracy, which is about 20% higher than the state-of-the-art parking behavior prediction method.

## Run the code

1. We use Anaconda to create a python 3.6 environment. 
2. run ``` pip install -r requirements.txt ``` to install package dependencies.
3. check the data path (line 26 and line 27) and other parameters in ```main.py```.
4. run ``` python main.py ``` to get the results.

We recommend that you use the jupyter notebook to visually view the experimental results.


## ThsParking
Our proposed periodic LSTM with weather-aware gating mechanism, PewLSTM, has been integrated into a real park- ing system, ThsParking （http://www.thsparking.com） (developed by Huaching Tech http://www.huaching.com/), which is one of the top smart parking platforms in China. Based on the prediction, the parking system can launch a segmented pricing strategy to utilize parking spaces better and gain more profit.

## Datasets

#### Parking datasets

Our parking datasets are composed of parking records from 10 parking lots in China, including shopping malls, hotels, communities, and so on. We list the total number of parking spaces, parking records, surroundings,and location for each parking lot in Table 1. The dataset spans 23 months from October 16th, 2017 to August 30th, 2019, which consists of 452,480 parking records in 683 days. For each record, we obtain the parking information including the arrival time, the departure time, date, parking space, and the price. Please note that the parking records with a duration of less than ﬁve minutes are regarded as noise data. The related weather dataset includes the hourly weather data from four districts where the 10 parking lots are built. For each hour, we collect the related weather information including temperature, wind speed, precipitation, and relative humidity. 

Please note that the records in the table are valid records with parking duration more than five minutes, and the actual total number of records is slightly more than the records in the Table 1.

**Table1**

| ParkingLot | Space\# | Record\# | Surrounding       | Location\(district/city/province\) |
|------------|---------|----------|-------------------|------------------------------------|
| P1         | 23      | 22,388   | hotel             | Haishu, Ningbo, Zhejiang           |
| P2         | 87      | 208,472  | commercial street | Haishu, Ningbo, Zhejiang           |
| P3         | 9       | 31,187   | shopping mall     | Yinzhou, Ningbo, Zhejiang          |
| P4         | 62      | 4,335    | industrial park   | Yinzhou, Ningbo, Zhejiang          |
| P5         | 46      | 28,804   | hotel             | Yinzhou, Ningbo, Zhejiang          |
| P6         | 16      | 32,009   | market            | Yuelu, Changsha, Hunan             |
| P7         | 31      | 49,707   | market            | Yuelu, Changsha, Hunan             |
| P8         | 27      | 46,365   | shopping mall     | Yuelu, Changsha, Hunan             |
| P9         | 49      | 5,808    | shopping mall     | Yuhua, Changsha, Hunan             |
| P10        | 65      | 23,405   | community         | Yuhua, Changsha, Hunan             |

#### Weather datasets

Our weather dataset comes from the public data released by the local meteorological station. The filename is the index of weather table. Index 0 is The weather information of Haishu, Ningbo, Zhejiang, Index 1 for Yinzhou, Ningbo, Zhejiang, Index 2 for Changsha, Hunan.

## Acknowledgement

PewLSTM is developed by Renmin University, Tsinghua University, Beijing University of Posts and Telecommunications,  Technische Universität Berlin,  National University of Singapore and  Peking University.

Feng Zhang, Yani Liu and Xiaoyong Du are with the Key Laboratory of Data Engineering and Knowledge Engineering (MOE), and School of Information, Renmin University of China, Beijing, 100872, China.

Ningxuan Feng and Jidong Zhai are with the Department of Computer Science and Technology, Tsinghua University, Beijing, 100084, China.

Shuhao Zhang is with the School of Computing, National University of Singapore, 119077, Singapore and Technische Universität Berlin, Berlin, 10623, Germany. 

Bingsheng He is with the School of Computing, National University of Singapore, 119077, Singapore.

Jiazao Lin is with the Department of Information Management, Peking University,  beijing, 100871, China. 

If you have problems, please contact:

* [whitycatty@gmail.com](whitycatty@gmail.com)
* [liuyn1999@gmail.com](liuyn1999@gmail.com)



## Citation
If you use our code or the data, please cite our paper:
```
@article{Zhang2020PewLSTM
  title={PewLSTM: Periodic LSTM with Weather-Aware Gating Mechanism for Parking Behavior Prediction}
  author={Feng Zhang, Ningxuan Feng, Yani Liu, Cheng Yang, Jidong Zhai, Shuhao Zhang, Bingsheng He, Jiazao Lin, Xiaoyong Du}
  conference={International Joint Conference on Artificial Intelligence(IJCAI)}
  year={2020}
}
```

Thanks for your interests in PewLSTM and hope you like it. (^_^)
