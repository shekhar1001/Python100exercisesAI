def is_armstrong(n):
    return n == sum(int(digit) ** len(str(n)) for digit in str(n))

# Example
num = 370
print(is_armstrong(num))