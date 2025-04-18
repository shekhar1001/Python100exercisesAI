import spacy
nlp=spacy.load("en_core_web_sm")

doc=nlp("Running runners ran easily")

for token in doc:
    print(f"{token.text} -> {token.lemma_}")

# Output:
# Running -> run
# runners -> runner
# ran -> run
# easily -> easily