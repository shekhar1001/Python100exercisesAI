from textblob import TextBlob

text= "I absolutely love this Movie depsite the story being boring"
blob=TextBlob(text)

print("Sentiment Polarity:", blob.sentiment.polarity)

