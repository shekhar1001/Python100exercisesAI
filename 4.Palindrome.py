def is_palindrome(s):
 if s==s[::-1]:
  print(s,"is a palindrome")
 else:
  print(s,"is not a palindrome")

string="madam"
string2="hello"

is_palindrome(string)
is_palindrome(string2)
