# Creating a simple line plot using matplotlib
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[10,20,25,40]

plt.plot(x,y, marker='o')
plt.title('Line plot example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()


# Creating simple bar chart using seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df=pd.DataFrame({
    'Category':['A','B','C','D'],
    'Value':[10,20,15,25]
})

sns.barplot(x='Category',y='Value',data=df)
sns.set_title('Bar plot example')
plt.show()
