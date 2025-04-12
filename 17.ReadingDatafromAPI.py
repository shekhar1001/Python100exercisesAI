import requests 

# Getting data from API(Public)
url='https://api.agify.io/?name=michael'
response=requests.get(url)

data=response.json()
print(data)