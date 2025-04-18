import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, "->", ent.label_)
# Output
# Apple -> ORG
# U.K. -> GPE
# $1 billion -> MONEYtokenization, word embeddings, text classification, sentiment analysis, and vectorization techniques
