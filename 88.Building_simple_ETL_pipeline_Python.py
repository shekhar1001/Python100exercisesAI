# Simple ETL:Extract, Transform, and Load
import pandas as pd

data={
    'name':["Shekhar", "Sudeep", "Suzata", "Tulasa"],
    'age':["27", "39","37", "23"],
    "salary": [50000, 100000,70000, 40000]
}
df=pd.DataFrame(data)

df['salary_in_euro']=df['salary']/156

df.to_csv('processed_data.csv', index=False)
