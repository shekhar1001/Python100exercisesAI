from sklearn.feature_extraction.text import TfidfVectorizer

corpus=("NLP ist eine aufregende Domäne","Machinelles lernen verwendet NLP")
tfidf=TfidfVectorizer()

X=tfidf.fit_transform(corpus)

print("TF-IDF-Matrix:\n",X.toarray())


#TF-IDF-Matrix:
#  [[0.47107781 0.47107781 0.47107781 0.47107781 0.         0.
#   0.33517574 0.        ]
#  [0.         0.         0.         0.         0.53404633 0.53404633
#   0.37997836 0.53404633]]