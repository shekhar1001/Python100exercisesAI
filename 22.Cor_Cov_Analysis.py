import pandas as pd

df=pd.DataFrame({
    'X':[1,2,3,4,5],
    'Y':[2,4,6,8,10],
    'Z':[3,4,6,3,1]
})

print("Correlation:\n",df.corr())

print("Covariance:\n",df.cov())