import pandas as pd
from textblob import TextBlob

# Load dataset
df = pd.read_csv("Tweets.csv")

# Show column names (just to verify)
print("Columns:", df.columns)

# Sentiment function
def get_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# APPLY sentiment analysis
df["Predicted_Sentiment"] = df["text"].apply(get_sentiment)  # change 'text' if needed

# Display results
print("\nSample Output:")
print(df[[df.columns[0], "Predicted_Sentiment"]].head())

print("\nSentiment Count:")
print(df["Predicted_Sentiment"].value_counts())

df.to_csv("Sentiment_Output.csv", index=False)
