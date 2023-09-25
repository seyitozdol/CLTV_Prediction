# **************************
# Business Problem:
# FLO wishes to define a roadmap for its sales and marketing activities. In order for the company
# to plan for the medium to long term, it's essential to estimate the potential value that existing customers will provide to the company in the future."
# **************************

# **************************
# Dataset Story:
# The dataset consists of information obtained from the past shopping behaviors of customers who shopped at FLO
# through OmniChannel (both online and offline) in the years 2020-2021."
# **************************

# **************************
# VARIABLES
# master_id: Unique customer number
# order_channel: The channel used for shopping, indicating the platform (Android, ios, Desktop, Mobile)
# last_order_channel: The channel where the most recent shopping was done
# first_order_date: The date of the customer's first purchase
# last_order_date: The date of the customer's last purchase
# last_order_date_online: The date of the customer's last purchase on an online platform
# last_order_date_offline: The date of the customer's last purchase on an offline platform
# order_num_total_ever_online: Total number of purchases the customer has made on an online platform
# order_num_total_ever_offline: Total number of purchases the customer has made offline
# customer_value_total_ever_offline: Total amount spent by the customer on offline purchases
# customer_value_total_ever_online: Total amount spent by the customer on online purchases
# interested_in_categories_12: List of categories the customer has shopped in over the last 12 months
# **************************

# **************************
# Project Tasks
# Task 1: Data Prepration
# Task 2: Creation of the CLTV Data Structure
# Task 3: Establishment of BG/NBD and Gamma-Gamma Models and Calculation of CLTV
# Task 4: Segmentation Based on CLTV Value


import datetime as dt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 500)
# pd.set_option('display_max_rows',None)
pd.set_option('display.float_format',lambda x : '%.5f' % x)

df_ = pd.read_csv('flo_data_20K.csv')
df = df_.copy()


def analyze_missing_values(df):
    na_cols = df.columns[df.isna().any()].tolist()
    total_missing = df[na_cols].isna().sum().sort_values(ascending=False)
    percentage_missing = ((df[na_cols].isna().sum() / df.shape[0]) * 100).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Count': total_missing, 'Percentage (%)': np.round(percentage_missing, 2)})
    return missing_data


# to get an initial understanding of the data's structure, its content, and if there are any missing values that need to be addressed.
def sum_df(dataframe, head=6):
    print("~~~~~~~~~~|-HEAD-|~~~~~~~~~~ ")
    print(dataframe.head(head))
    print("~~~~~~~~~~|-TAIL-|~~~~~~~~~~ ")
    print(dataframe.tail(head))
    print("~~~~~~~~~~|-TYPES-|~~~~~~~~~~ ")
    print(dataframe.dtypes)
    print("~~~~~~~~~~|-SHAPE-|~~~~~~~~~~ ")
    print(dataframe.shape)
    print("~~~~~~~~~~|-NUMBER OF UNIQUE-|~~~~~~~~~~ ")
    print(dataframe.nunique())
    print("~~~~~~~~~~|-NA-|~~~~~~~~~~ ")
    print(dataframe.isnull().sum())
    print("~~~~~~~~~~|-QUANTILES-|~~~~~~~~~~ ")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("~~~~~~~~~~|-NUMERIC COLUMNS-|~~~~~~~~~~ ")
    print([i for i in dataframe.columns if dataframe[i].dtype != "O"])
    print("~~~~~~~~~~|-MISSING VALUE ANALYSIS-|~~~~~~~~~~ ")
    print(analyze_missing_values(dataframe))

sum_df(df)

