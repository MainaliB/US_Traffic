import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler



######################################################################################################    



def dropping_nulls(df, columns, proportion_column):
    '''This function given the dataframe, the columns, and the proportion for column returns a 
    dataframe with the columns with the proportion of null values higher than the given proportion dropped'''
    for col in columns:
        if (df[col].isnull().sum())/len(df) > proportion_column:
            df.drop(columns = [col], inplace = True)
    return df

######################################################################################################    



def data_conversion(df):
    '''This function given the dataframe converts all of the boolean datatypes to integer data type
    and return the dataframe'''

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df

######################################################################################################    



def clean_data(df):
    '''This function given the dataframe does all the preparation and returns a prepared data for exploration'''

    df = df.drop(columns = ['Unnamed: 0'])
    df.columns = [col.lower() for col in df.columns]
    df = df.dropna()
    df = df.reset_index().drop(columns = ['index'])
    df['start_time'] = pd.to_datetime(df.start_time)
    df['weekday'] = df['start_time'].apply(lambda x: x.weekday())
    df['end_time'] = pd.to_datetime(df.end_time)
    df['duration'] = (df.end_time-df.start_time )/pd.Timedelta('1h')
    df = data_conversion(df)

    return df

######################################################################################################    



def scaler(train, test, validate, cols_to_ignore):
    train_to_scale = train.drop(columns = cols_to_ignore)
    validate_to_scale = validate.drop(columns = cols_to_ignore)
    test_to_scale = test.drop(columns = cols_to_ignore)
    
    # create a scaler object
    scaler = MinMaxScaler()

    # fit our scaler object to our train dataframe created to scale
    scaler = scaler.fit(train_to_scale)
    
    # get our scaled dataframes
    train_scaled = pd.DataFrame(scaler.transform(train_to_scale), columns = train_to_scale.columns,
                                index = train_to_scale.columns)
    
    validate_scaled = pd.DataFrame(scaler.transform(validate_to_scale), columns = validate_to_scale.columns,
                                index = validate_to_scale.columns)
    
    test_scaled = pd.DataFrame(scaler.transform(test_to_scale), columns = test_to_scale.columns,
                                index = test_to_scale.columns)
    
    return train_scaled, test_scaled, validate_scaled

    
######################################################################################################    
    
    
def prep_data(df, cols_to_ignore):
    df_s1 = df[df['severity']==1]
    df_s2 = df[df['severity']==2]
    df_s3 = df[df['severity']==3]
    df_s4 = df[df['severity']==4]

    # lets get the count of each dataframe and select the count of the first column and then get the max count to get
    # the count to upsample to
    count = max(df_s1.count()[0], df_s2.count()[0], df_s3.count()[0], df_s4.count()[0])
   
    #upsampling
    df_s1 = resample(df_s1, replace=df_s1.count()[0]<count, n_samples=count, random_state=42)
    df_s2 = resample(df_s2, replace=df_s2.count()[0]<count, n_samples=count, random_state=42)
    df_s3 = resample(df_s3, replace=df_s3.count()[0]<count, n_samples=count, random_state=42)
    df_s4 = resample(df_s4, replace=df_s4.count()[0]<count, n_samples=count, random_state=42)
    
    # adding all of the dataframes together
    df = pd.concat([df_s1, df_s2, df_s3, df_s4])
    
    # lets split our data
    train, validate = train_test_split(df, random_state = 123, test_size = .30)
    test, validate = train_test_split(validate, random_state = 121, test_size = .50)
    
    #lets scaled our data
    train_scaled, test_scaled, validate_scaled = scaler(train, test, validate, cols_to_ignore)
    
    return train_scaled, test_scaled, validate_scaled



######################################################################################################    


def getting_top_features(train_scaled, y_train, k):
    
    f_selector = SelectKBest(f_regression, k=k)

    f_selector.fit(train_scaled, y_train)

    feature_mask = f_selector.get_support()

    f_feature = train_scaled.iloc[:,feature_mask].columns.tolist()
    
    return f_feature
    




