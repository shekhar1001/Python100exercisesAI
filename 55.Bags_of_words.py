from sklearn.feature_extraction.text import CountVectorizer
corpus=["Ich liebe Machinelleslernen","Machinelleslernen ist leistungsstark"]
vectorizer=CountVectorizer()

X=vectorizer.fit_transform(corpus)

print("Vokabular:",vectorizer.vocabulary_)
print("Bow-Matrix:\n",X.toarray())
print("Feature-Namen:",vectorizer.get_feature_names_out)

# Vokabular: {'ich': 0, 'liebe': 3, 'machinelleslernen': 4, 'ist': 1, 'leistungsstark': 2}
# Bow-Matrix:
#  [[1 0 0 1 1]
#  [0 1 1 0 1]]
# Feature-Namen: <bound method CountVectorizer.get_feature_names_out of CountVectorizer()>