import requests
from bs4 import BeautifulSoup
import pandas as pd

# Web scraping example using BeautifulSoup
url='https://www.python.org'
response=requests.get(url)

soup=BeautifulSoup(response.text,'html.parser')
title=soup.title.text
print('Website Title',title)