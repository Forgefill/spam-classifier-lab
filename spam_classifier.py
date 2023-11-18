import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('spam.csv', encoding='UTF-8')
print(data.head())

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v2": "text", "v1": "label"})

# Remove punctuation and stopwords from text
def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

data['text'] = data['text'].apply(text_process)

# Print data information
print("\nData between rows 1990 and 2000:")
print(data)
print("\nLabel value counts:")
print(data['label'].value_counts())


# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['text'])
print("Vector shape:", vectors.shape)

X_train, X_test, y_train, y_test = train_test_split(vectors, data['label'], test_size=0.15, random_state=111)

mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
rfc = RandomForestClassifier(n_estimators=31, random_state=111)

# Create a dictionary of models
clfs = {'NB': mnb, 'DT': dtc, 'RF': rfc}

# Function to train a classifier
def train(clf, features, targets):
    clf.fit(features, targets)

# Function to make predictions
def predict(clf, features):
    return clf.predict(features)

# Function to calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, pos_label='spam')
    recall = recall_score(true_labels, predicted_labels, pos_label='spam')
    f1 = f1_score(true_labels, predicted_labels, pos_label='spam')
    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=['ham', 'spam'])
    
    return precision, recall, f1, confusion_mat

# Fit the data onto the models and calculate accuracy, precision, recall, and f1 score
pred_scores_word_vectors = []
for k, v in clfs.items():
    train(v, X_train, y_train)
    pred = predict(v, X_test)
    
    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, confusion_mat = calculate_metrics(y_test, pred)
    
    pred_scores_word_vectors.append((k, [accuracy, precision, recall, f1, confusion_mat]))
    print(f"{k}:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nConfusion Matrix:\n{confusion_mat}\n")

