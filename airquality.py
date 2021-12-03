import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import plotly.figure_factory as ff
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

st.markdown("## Understanding the effect of Covid-19 lockdown on air quality  ")
st.markdown("The World Air Quality Index project team has been taking measurements from stations planted in different cities around the world. In this project, We'll be interested only in the years 2019, 2020, and 2021. Within the dataset, we'll find the min, max, median, and standard deviation of the measurements for each of the air pollutant species (PM2.5, PM10, Ozone ...).")

france = pd.read_csv('france_clean.csv')
france['Date'] = pd.to_datetime(france['Date'])
france = france.set_index('Date')
france = france[france["City"]=="Amiens"]

st.markdown("## Dataset ")
# st.dataframe(france)
if st.checkbox('Dataset'):
    st.write(france)
st.markdown("### Calculating the AQI ")
## PM10 Sub-Index calculation
def get_pm10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

france["pm10_SubIndex"] = france["pm10"].apply(lambda x: get_pm10_subindex(x))


## PM2.5 Sub-Index calculation
def get_pm25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

france["pm25_SubIndex"] = france["pm25"].apply(lambda x: get_pm25_subindex(x))

## O3 Sub-Index calculation
def get_o3_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

france["o3_SubIndex"] = france["o3"].apply(lambda x: get_o3_subindex(x))

## NO2 Sub-Index calculation
def get_no2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

france["no2_SubIndex"] = france["no2"].apply(lambda x: get_no2_subindex(x))

## CO Sub-Index calculation
def get_co_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

france["co_SubIndex"] = france["co"].apply(lambda x: get_co_subindex(x))

france = france[["pm10_SubIndex","pm25_SubIndex","o3_SubIndex","no2_SubIndex","co_SubIndex","type"]]
st.text(" \n")
st.text(" \n")
if st.checkbox('AQI calculations'):
    st.write(france)


## AQI bucketing
def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return np.NaN

france["Checks"] = (france["pm10_SubIndex"] > 0).astype(int) + \
                (france["pm25_SubIndex"] > 0).astype(int) + \
                (france["o3_SubIndex"] > 0).astype(int) + \
                (france["no2_SubIndex"] > 0).astype(int) + \
                (france["co_SubIndex"] > 0).astype(int) 
                

france["AQI_calculated"] = round(france[["pm10_SubIndex", "pm25_SubIndex", "o3_SubIndex", "no2_SubIndex",
                                 "co_SubIndex"]].max(axis = 1))
france.loc[france["pm25_SubIndex"] + france["pm10_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
france.loc[france.Checks < 3, "AQI_calculated"] = np.NaN

france["AQI_Class"] = france["AQI_calculated"].apply(lambda x: get_AQI_bucket(x))
st.text(" \n")
st.text(" \n")
st.markdown("### AQI calculated")
if st.checkbox('AQI Values'):
    st.write(france[~france.AQI_calculated.isna()])
# st.dataframe(.head(13))

france[["AQI_calculated"]].sort_values(by='Date',ascending=False)
france[["AQI_calculated"]].plot(figsize=(15, 6))
plt.show()
y=france[["AQI_calculated"]].AQI_calculated

france = france.drop(['Checks'],axis=1)

france.AQI_Class.value_counts()

france['AQI_Class'] = france['AQI_Class'].map({'Good': 5, 'Satisfactory': 4, 'Moderate': 3, 'Poor': 2,'Very Poor': 1, 'Severe': 0})

france_uni_var = france[["AQI_calculated"]]

france = france.dropna()
st.text(" \n")
st.text(" \n")
# st.dataframe(france)

train_len = 800
train = france_uni_var[0:train_len] 
test = france_uni_var[train_len:] 

y_hat_naive = test.copy()
y_hat_naive['naive_forecast'] = train['AQI_calculated'][train_len-1]

# st.plt.figure(figsize=(20,5))
# plt.grid()
# plt.plot(train['AQI_calculated'], label='Train')
# plt.plot(test['AQI_calculated'], label='Test')
# plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')
# plt.legend(loc='best')
# plt.title('Naive Method')
# plt.show()
# st.plotly_chart(france[["AQI_calculated"]], use_container_width=True)

st.bar_chart(train['AQI_calculated'].dropna())
st.bar_chart(test['AQI_calculated'].dropna())
# st.bar_chart(y_hat_naive['naive_forecast'].dropna())
# x1 = train['AQI_calculated'].dropna()
# x2 = test['AQI_calculated'].dropna()
# x3 = y_hat_naive['naive_forecast'].dropna()
# data = pd.DataFrame([x1, x2, x3])

# st.line_chart(data)
# # group_labels = ['Train', 'Test', 'Naive forecast']
# # fig = ff.create_distplot(hist_data, group_labels, bin_size=[.20, .25, .5])
# # st.bar_chart(fig, use_container_width=True)


st.text(" \n")
st.markdown("### Model Building ")
from sklearn.model_selection import train_test_split
feature_cols = ['pm10_SubIndex','pm25_SubIndex', 'o3_SubIndex',	'no2_SubIndex',	'co_SubIndex',	'type']
X = france[feature_cols].values # Features
y = france.AQI_Class.values # Target variable
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=False, random_state=42)
st.write('The number of training samples: {}\nThe number of testing samples: {}'.format(X_train.shape[0], X_test.shape[0]))