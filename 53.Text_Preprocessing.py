import string

text="Ich heiße Shekhar, erinnern?"

text=text.lower()

text=text.translate(str.maketrans('','',string.punctuation))

tokens=text.split()

print("Tokens:",tokens)
