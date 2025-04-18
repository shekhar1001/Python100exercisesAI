from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

#Training data (in German)
data = ["Ich liebe diesen Film", "Dieser Film ist schrecklich", "Ich mag diesen Film nicht"]
labels = [1, 0, 0]  #1 = Positive, 0 = Negative

#Creating Pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(data, labels)

print("Prediction:", model.predict(["Ich hasse diesen Film"]))  #Expected: 0 (Negative)


#Prediction: [0]