import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# User input
news = input("Enter News Text:\n")

# Convert text into numbers
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

# Output
if prediction[0] == 1:
    print("\nREAL NEWS")
else:
    print("\nFAKE NEWS")