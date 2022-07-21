# import Libraries
import pandas as pd
import numpy as np

# Import Test data for predictions
df_test = pd.read_csv(r'C:\Users\OlumayowaOyaleke\Desktop\Anomaly Detection\ML-MATT-CompetitionQT1920_test.csv',encoding='windows-1252')

# Preprocess data (Getting Test Dataset in the same format as training dataset)
def preprocess_data(df):
    
    #Parse TimeStamp
    # df.Time = pd.to_datetime(df.Time.str.lower(), format="%H:%M")
    df.Time = pd.to_datetime(df.Time, format="%H:%M")
    
    #Convert Unusual Column to Object Data Type
    # df.Unusual = df.Unusual.astype('object')
    
    #Convert maxUE_DL Column to Object Data Type
    df.maxUE_DL = df.maxUE_DL.astype('object')

    #Convert maxUE_UL Column to Object Data Type
    df.maxUE_UL = df.maxUE_UL.astype('object')

    #Convert maxUE_UL+DL Column to Object Data Type
    df['maxUE_UL+DL'] = df['maxUE_UL+DL'].astype('object')
    
    #Convert Object Data Type to Category
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label] = content.astype("category").cat.as_ordered()
            
     #Fill in Numerical Columns with the median
            if pd.api.types.is_float_dtype(content):
                 if pd.isnull(content).sum():
                    df[label] = content.fillna(content.median())
                    
     
      #Fill in Categorical Columns with the mode
            if pd.api.types.is_categorical_dtype(content):
                if pd.isnull(content).sum():
                    print(label)
                    df[label] = content.fillna(content.value_counts().index[0]) 
                    
        # Feature Enginering on Date Column             
        df['Year'] = df.Time.dt.year
        df['Month'] = df.Time.dt.month
        df['Day'] = df.Time.dt.day
        df['Hour'] = df.Time.dt.hour
        df['Min'] = df.Time.dt.minute
        df['Seconds'] = df.Time.dt.second
        
        # Drop Year, Month, Day and Second Column
        df.drop('Year', axis = 1,inplace = True)
        df.drop('Month', axis = 1, inplace = True)
        df.drop('Day', axis = 1, inplace = True)
        df.drop('Seconds', axis = 1, inplace = True)
           
          
        # Encoding Categorical Columns    
        for label, content in df.items():
            if not pd.api.types.is_numeric_dtype(content):
                # We add the +1 because pandas encodes missing categories as -1
                df[label] = pd.Categorical(content).codes+1    
                
        df.drop('Time',axis = 1, inplace = True)         
    return df

    preprocess_data(df_test)

    df_test
    