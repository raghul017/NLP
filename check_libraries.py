# import nltk 
# nltk.download('punkt') # download the punkt tokenizer model for splitting text into tokens
# from nltk.tokenize import word_tokenize

# text = ' I love learning NLP . It is so exciting and powerful !'
# tokens = word_tokenize(text)
# print(tokens)


# import spacy
# nlp = spacy.load("en_core_web_sm")

# doc = nlp("Elon Musk founded Tesla in California in 2003.")
# for ent in doc.ents:
#     print(ent.text, ent.label_)


# from textblob import TextBlob

# text = "I love learning NLP. It's so exciting and powerful!"
# analysis = TextBlob(text)
# print("Sentiment Score:", analysis.sentiment.polarity)

# from transformers import pipeline

# # Load a sentiment analysis pipeline (BERT model)
# classifier = pipeline("sentiment-analysis")

# # Test on some text
# text = "I absolutely love NLP and AI!"
# result = classifier(text)

# print(result)  # Output: [{'label': 'POSITIVE', 'score': 0.9998}]

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# def sentiment_analysis():
#     while True:
#         text = input("Enter a text to analyze: ")
#         if text.lower() == "exit":
#             print("Exiting the program...")
#             break
#         result = classifier(text)
#         print(result)
        
# sentiment_analysis()

reviews = [
    "I love this product! It's amazing and I highly recommend it.",
    "This is the worst thing I've ever bought. It's not worth the money.",
    "I'm not sure I can recommend this product. It's okay, but not great.",
    "I'm very happy with my purchase. It's exactly what I needed.",
]

results = classifier(reviews)

for review, result in zip(reviews, results):
    print(f"Review: {review}")
    print(f"Sentiment: {result['label']} (Score: {result['score']:.2f})")
    print()
    
