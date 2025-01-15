import nltk
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# NLTK Resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")


# 1. Data pre-processing

# Downloading dataset
data_path = "dataset/JobLevelData.xlsx"

data = pandas.read_excel(data_path)

data.fillna("", inplace=True) # Replacing NaN > empty string

# Labels processing
data["Combined_Labels"] = data[["Column 1", "Column 2", "Column 3", "Column 4"]].values.tolist() # Split the labels
data["Combined_Labels"] = data["Combined_Labels"].apply(lambda x: list(filter(None, x)))  # Remove empty values

def text_cleaner(text):
    text = text.lower() # Register
    tokens = nltk.word_tokenize(text) # Tokenization
    stop_words = set(stopwords.words("english")) # Noise with NLTK

    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)
    tokens = filtered_tokens

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word in tokens:
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_tokens.append(lemmatized_word)
    tokens = lemmatized_tokens

    return " ".join(tokens)

# Title processing
data["Processed_Title"] = data["Title"].apply(text_cleaner)

# attributes X abd labels Y
X_titles = data["Processed_Title"]
Y_labels = data["Combined_Labels"]

# Labels to binary
binar = MultiLabelBinarizer()
y_binarized = binar.fit_transform(Y_labels)


# 2. TF-IDF vectorization
vector = TfidfVectorizer()
X_tfidf = vector.fit_transform(X_titles)


# 3. 80/20 data split
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, y_binarized, test_size=0.20, random_state=1)


# 4. Training
model = RandomForestClassifier(random_state=1, n_estimators=100)
model.fit(X_train, Y_train)

# 5. Result
Y_pred = model.predict(X_test)

results = classification_report(Y_test, Y_pred, target_names=binar.classes_, zero_division=0)
accuracy = accuracy_score(Y_test, Y_pred)

# Print metrics
print(f"Classification Report:\n{results}")
print(f"Total model accuracy:\n{accuracy}")