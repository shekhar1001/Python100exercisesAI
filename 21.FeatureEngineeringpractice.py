import pandas as pd

df=pd.DataFrame({
    'Name':['Shekhar','Sujata',],
    "DOB":['1998','1987']
})

current_year=2025

df['DOB']=df['DOB'].astype(int)

df['Age']=current_year-df['DOB']

# Converting categorical string into number

df['Is_Shekhar']=df['Name'].apply(lambda x:1 if x== 'Shekhar' else 0)

print(df)

