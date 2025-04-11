import pandas as pd

data={
    'Department':['HR','IT','HR','Finance','IT'],
    'Salary':[3500,5000,4000,4000,6000]
}
df=pd.DataFrame(data)
grouped=df.groupby('Department')['Salary'].mean()
print(grouped)
print(grouped.sum())
