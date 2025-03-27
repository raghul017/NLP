# import nltk 
# import re
# import string
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# #Download necessary NLTK resources
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

# #Sample text 
# text = "NLP is amazing!!! It's evolving in 2025. I love working with AI & NLP."

# def preprocess_text(text):

#     #lowercase
#     text = text.lower()

#     #remove punctuation
#     text= text.translate(str.maketrans("","" , string.punctuation))

#     #remove numbers
#     text = re.sub(r'\d+' , '' , text)

#     #tokenize words
#     tokens = word_tokenize(text)

#     #remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]

#     #Lemmetization
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]

#     return " ".join(tokens)

# #run preprocessing
# cleaned_text = preprocess_text(text)
# print("Original text: ", text)
# print("Cleaned text: ", cleaned_text)
    


#POS tagging and Named Entity Recognition

import nltk
import spacy
from nltk.tokenize import word_tokenize

nltk.download('average_perceptron_tagger')

text = "Elon musk is the CEO of Tesla. He founded SpaceX in 2002."

#tokenize text
tokens = word_tokenize(text)

#apply pos tagging
pos_tag = nltk.pos_tag(tokens)

print("POS Tags: ", pos_tag)

#Named entity recognition 
nlp = spacy.load("en_core_web_sm")

#process text 
doc = nlp(text)

#print named entities
for ent in doc.ents:
    print(f'Entity : {ent.text} , Type: {ent.label_}')