import numpy as mynp
my_l1 = [113,213,313,413,567]
print(f'my_l1 type is: {type(my_l1)}')
mynd_arr = mynp.array(my_l1)
print('mynd_arr type is: --> '+
str(type(mynd_arr)))
print('mynd_arr -->'+ str(mynd_arr))
print('The Array dimensions is: '+
str(mynd_arr.ndim))
print('The data type of array elements is :  '+
str(mynd_arr.dtype))
print('The array size is : '+ str(mynd_arr.size))
print('The array shape is : '+
str(mynd_arr.shape))