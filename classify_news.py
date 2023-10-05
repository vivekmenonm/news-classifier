import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer

# Text preprocessing and vectorization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)


# Load your news dataset (Assuming it's in a CSV file)
df = pd.read_csv('news_data.csv')

# Split dataset into training and testing sets
X = df['news_article']
y = df['news_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text preprocessing and vectorization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)

X_train = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word.isalnum()]) for text in X_train]
X_test = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word.isalnum()]) for text in X_test]

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to load the model and make predictions
def predict_category(input_text):
    # Load the saved model
    saved_model = joblib.load('news_classifier_model.pkl')
    
    # Preprocess the input text
    input_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(input_text.lower()) if word.isalnum()])
    
    # Vectorize the input text
    input_text_tfidf = vectorizer.transform([input_text])
    
    # Predict the category
    predicted_category = saved_model.predict(input_text_tfidf)[0]
    
    return predicted_category

# Example usage
# input_text = "US Federal Communications Commission Chairman Ajit Pai has announced that he will step down from his position on January 20. He will be leaving office on the same day as President-elect Joe Biden's inauguration. Previously, former FCC Chairman Tom Wheeler had left office on 20th January 2017, the day President Donald Trump was sworn in."
# predicted_category = predict_category(input_text)
# print("Predicted Category:", predicted_category)