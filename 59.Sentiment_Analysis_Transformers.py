from transformers import pipeline

# Loading German sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

text = "Ich liebe diesen Film weil er so spannend ist"
result = classifier(text)

# Output
print("Label:", result[0]['label'])
print("Confidence:", round(result[0]['score'], 2))


# Label: positive
# Confidence: 0.98