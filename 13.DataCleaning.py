import pandas as pd
import numpy as np

data={
    'Name':['Shekhar','Sujata','Sudeep'],
    'Age':[25,np.nan,38]
}
df=pd.DataFrame(data)

df['Age'].fillna(df['Age'].mean(),inplace=True)

print(df)

print(df.isnull().sum)