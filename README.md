# Winter25 ECE143 Project
# Predicting Used Phones & Tablets Prices
***
Group3:
Eric Gu, HongjieWang, Hesam Mojtahedi , Ching-Hao Wang, Chainika Shah
***
## Introduction

The used and refurbished device market has grown significantly in recent years due to its cost-effectiveness for both consumers and businesses. However, pricing these devices accurately remains a challenge. Incorrect pricing can lead to financial losses for sellers if the price is too low or poor sales performance if the price is too high. Consumers also face uncertainty when determining whether they are getting a good deal. Therefore, there is a need for a data-driven approach to predict used phone and tablet prices accurately.

***
## About Dataset 
- The [Data Exploratory Analysis Demo](https://wi25-ece143-team3.streamlit.app/) we deployed via streamlit:
  ![image](https://github.com/user-attachments/assets/624d6d1a-e1e5-46c9-80ec-7e89d34e36ce)

- [Dataset Link](https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data/data)
- CONTEXT- The used and refurbished device market has grown considerably over the past decade as it provide cost-effective alternatives to both consumers and businesses that are looking to save money when purchasing one. Maximizing the longevity of devices through second-hand trade also reduces their environmental impact and helps in recycling and reducing waste. Here is a sample dataset of normalized used and new pricing data of refurbished / used devices.

| Column Name            | Description                                           |
|------------------------|-------------------------------------------------------|
| device_brand          | Name of manufacturing brand                           |
| os                   | OS on which the device runs                            |
| screen_size          | Size of the screen in cm                               |
| 4g                   | Whether 4G is available or not                         |
| 5g                   | Whether 5G is available or not                         |
| front_camera_mp      | Resolution of the front camera in megapixels          |
| back_camera_mp       | Resolution of the rear camera in megapixels           |
| internal_memory      | Amount of internal memory (ROM) in GB                 |
| ram                 | Amount of RAM in GB                                    |
| battery             | Energy capacity of the device battery in mAh          |
| weight              | Weight of the device in grams                          |
| release_year        | Year when the device model was released                |
| days_used           | Number of days the used/refurbished device has been used |
| normalized_new_price | Normalized price of a new device of the same model     |
| normalized_used_price (TARGET) | Normalized price of the used/refurbished device |

## Prediction Approach
- Linear Regression
- K-Nearest-Neighbors model
- Neural Network model
- ...

## Set-Up Commands after Cloneing the Repo
- With Python3.9+, to install all the dependent libraries:
```shell
pip install -r requirements
```
- To run streamlit app locally:
```shell
streamlit run streamlit_app.py
```

## Project Stage
1. Extracting and cleaning up data
2. Exploratory Data Analysis
3. Data visualization 
4. Predictor Approach & Final Presentation
