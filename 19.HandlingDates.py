import pandas as pd
from datetime import datetime

# Creating a dataframe with data strings

data=({
    'date':['2024-04-12','2024-04-07','2024-04-10']
})

data['date']=pd.to_datetime(data['date'])

data['days_since']=(datetime.now()-data['date']).days

print(data)