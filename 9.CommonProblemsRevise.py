# Basic Data Types & Operations
a=10 #Integer
b=3.14 #Float
c="Hello" #string
d=True #Boolean

#Operations
sum=a+b
concat_string=c+"World"
logical_not=not d

print(sum,concat_string,logical_not)

# Loops & Conditionals
for i in range(1,11):
    if i%2==0:
        print(i)#Checks for even number

# Functions, Lambda, Map, Filter
def square(x):
    return x*x

# Using lambda function
squared_lambda=lambda x:x*x

#Using Map function
nums=[1,2,3,4,5]
squared=list(map(square,nums))
evens=list(filter(lambda x:x%2==0,nums))
print(squared,evens)

#List Comprehensions
#creating a list of squares
squares=[x**2 for x in range(1,6)]

# creating list of even numbers 1 to 10
evens=[x for x in range(1,11) if x%2==0]

print(squares,evens)

# File Handling

# Wrting a file
with open("sample.txt","w") as file:
    file.write("Hello There, Its a text file for you to read.")
# Reading a file
with open("sample.txt","r") as file:
          content=file.read()

print(content) 

#Exception Handling

try:
    result=10/0
except ZeroDivisionError:
    print("Can't divide by zero")
finally:
    print("This block always executes")

