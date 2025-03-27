import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer , WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

#stemming

stemmer = PorterStemmer()
text = "The cats are playing in the garden. They played all day."

tokens = word_tokenize(text)

#apply stemming
stemmed_words = [stemmer.stem(word) for word in tokens]
print("Stemmed Words: ", stemmed_words)

#lemmatization
lemmatizer = WordNetLemmatizer()

#apply lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

print("Lemmatized Words: " , lemmatized_words)
