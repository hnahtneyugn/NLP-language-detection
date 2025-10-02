import os
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


working_dir = os.path.expanduser("~/projects/NLP/NLP-language-detection/data/clean_data")


# Load train and test set from text
train_df = pd.read_csv(f"{working_dir}/train.txt", sep="\t", header=None, names=["text", "label"], quoting=csv.QUOTE_NONE)
test_df = pd.read_csv(f"{working_dir}/test.txt", sep="\t", header=None, names=["text"], quoting=csv.QUOTE_NONE)
test_labels = pd.read_csv(f"{working_dir}/test_labels.txt", header=None, names=["label"], quoting=csv.QUOTE_NONE)


# Shuffle training data
train_df = train_df.sample(frac=1, random_state=69).reset_index(drop=True)


# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=50000)

X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_labels['label']


# Train the model with SVM
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# Making predictions
y_pred = nb_model.predict(X_test)


# Evaluating model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Test some sentences
sentences = [
    "Bonjour, comment ça va ?",        
    "Xin chào, bạn khỏe không?",       
    "Hello, how are you?",             
    "Привет, как дела?",               
    "Guten Tag, wie geht es Ihnen?",   
    "مرحبا، كيف حالك؟",                 
    "ສະບາຍດີ, ເຈົ້າສະບາຍດີບໍ?"
]

# Transform sentences to TF-IDF vectors
X_new = vectorizer.transform(sentences)

# Predict labels
predictions = nb_model.predict(X_new)

# Show results with predicted labels
for sent, pred in zip(sentences, predictions):
    print(f"Input: {sent}\nPredicted language: {pred}\n")

