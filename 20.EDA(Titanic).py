import seaborn as sns
import pandas as pd

df=sns.load_dataset('titanic')

print(df.info())

print(df.isnull().sum())

sns.countplot(x='class',data=df)
sns.countplot(x='class',hue='survived',data=df)

