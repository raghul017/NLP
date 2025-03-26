from transformers import pipeline

# Load the sentiment analysis model
classifier = pipeline("sentiment-analysis")


#User Input
def analyze_sentiment():
    while True:
        text = input("Enter a sentence (or type 'exit' to stop): ")
        if text.lower() == 'exit':
            break
        result = classifier(text)
        print("Sentiment:", result[0]['label'], "| Score:", round(result[0]['score'], 2))

# Run the interactive sentiment analysis
analyze_sentiment()

#Batch Processing
reviews = [
    "I absolutely love this product! It's amazing.",
    "The service was terrible, and I’m never coming back.",
    "It was an okay experience, not great but not bad.",
    "The movie was a masterpiece! I would watch it again.",
    "I hated the food. It was the worst meal I've ever had."
]

# Analyze each review
results = classifier(reviews)

for review, sentiment in zip(reviews, results):
    print(f"Review: {review}\nSentiment: {sentiment['label']} | Score: {round(sentiment['score'], 2)}\n")


#saving to csv file
import pandas as pd 

results = classifier(reviews)

# Convert results into a Pandas DataFrame
df = pd.DataFrame({
    "Review": reviews,
    "Sentiment": [res['label'] for res in results],
    "Score": [res['score'] for res in results]
})

# Save to CSV
df.to_csv("sentiment_analysis_results.csv", index=False)

print("Sentiment analysis saved to 'sentiment_analysis_results.csv'!")