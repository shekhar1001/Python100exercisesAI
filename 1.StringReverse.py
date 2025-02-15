# Write a Python program to reverse a given string.
# Using String sclicing
def reverse_string(string1):
    return string1[::-1]

string1="Hello"
print(reverse_string(string1))

# Using a Loop

def reverse_string2(string):
    reverse_string2=""
    for i in string:
        reverse_string2=i+reverse_string2
        return reverse_string2

print(reverse_string2("Hello"))

# Using Recursion
def reverse_string3(string):
    if len(string)==0:
        return string
    else:
        return reverse_string(string[1:])+string[-1]
    
    