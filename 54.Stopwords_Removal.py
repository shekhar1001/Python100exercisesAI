import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

tokens=['ich', 'heiße', 'Shekhar','errinern']
filtered=[word for word in tokens if word not in stopwords.words('deutsch')]

print("Nach dem Entfernen des Stoppworts:",filtered)
