import numpy as mynp
from datetime import datetime
myarr_1 = mynp.array([14,15,16])
myarr_2 = mynp.array([11,12,13])
#conventional Python code
def my_dot_product(myarr_1,myarr_2):
    my_result = 0
    for _x,_y in zip(myarr_1,myarr_2):
        my_result +=  _x*_y
    return my_result
mybefore_time = datetime.now()
for myloop in range(2000000):
    my_dot_product(myarr_1,myarr_2)
myafter_time = datetime.now()
print('Time take to execute using conventional Python approach:',myafter_time-mybefore_time)
#code using numpy library
mybefore2_time = datetime.now()
for my_loop in range(2000000):
    mynp.dot(myarr_1,myarr_2) # using numpy
myafter2_time = datetime.now()
print('Time take to execute using Numpy Library:',myafter2_time-mybefore2_time)