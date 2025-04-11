import pandas as pd

employees=pd.DataFrame({
    'ID':[1,2,3],
    'Name':['Shekhar','Sujata','Sudeep']
})

salaries=pd.DataFrame({
    'ID':[1,2,3],
    'Salary':[4050,5000,6000]
})

merged=pd.merge(employees,salaries,on='ID')
print(merged)