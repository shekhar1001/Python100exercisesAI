import pandas as pd
s=pd.Series([1,2,3], index=['a','b','c'])

data={
    'Name':['Shekhar','Sujata'],
    'Age':[27,35]

}

df=pd.DataFrame(data)


print(s)
print(df)
