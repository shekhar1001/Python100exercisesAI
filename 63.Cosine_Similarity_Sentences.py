from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences=["Ich liebe Katzen","Ich vermisse meine Katzen", "Ich liebe auch Hunde"]
vec=TfidfVectorizer().fit_transform(sentences)

similarity=cosine_similarity(vec[0], vec[1])
print("Similarity between sentence 1 and 2:", similarity[0][0])

# Output
# Similarity between sentence 1 and 2: 0.44167133158789906