video_url = "https://www.youtube.com/watch?v=m9s1NQG3TNY&t=124s"

from googleapiclient.discovery import build
import pandas as pd

API_KEY = "AIzaSyAv5BNAjJ2jNXHhd-haKxkCS2FlTChGL3k"
VIDEO_ID = "j9JP5vKM1dc"  # just the ID, not full URL

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(video_id, max_comments=100):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_comments]

# Fetch and save
comments = get_comments(VIDEO_ID, max_comments=3000)
df = pd.DataFrame(comments, columns=["comment"])
df.to_csv("youtube_comments.csv", index=False)
df.head()

df

import pandas as pd

df=pd.read_csv("/content/dfcomments.csv")

df.to_csv("dfcomments.csv", index=False)

df.isnull().sum()

df.isnull().mean() * 100

df[df['comment'].isnull()]

df = df.dropna(subset=['comment']).reset_index(drop=True)

df.isnull().sum()

import re

def is_valid_comment(text):
    if not isinstance(text, str):
        return False
    text = text.strip()

    # Too short
    if len(text) < 10:
        return False

    # Contains links
    if 'http' in text or 'www.' in text or '.com' in text:
        return False

    # Only symbols/emojis
    if re.fullmatch(r'[\W_]+', text):
        return False

    # Not enough alphabetic characters
    if len(re.findall(r'[a-zA-Z]', text)) < len(text) * 0.4:
        return False

    return True

# Assuming you already loaded your full DataFrame as `df`
df = df.dropna(subset=['comment']).reset_index(drop=True)

# Apply your filter
filtered_df = df[df['comment'].apply(is_valid_comment)].reset_index(drop=True)

print(f"✅ Total valid comments: {len(filtered_df)}")
filtered_df.head()

!pip install googletrans==4.0.0-rc1

from googletrans import Translator

translator = Translator()

def translate_if_needed(text):
    try:
        if len(text.strip()) < 10:
            return text  # Skip very short comments

        # Use translation for comments with low % of English characters
        if len(re.findall(r'[a-zA-Z]', text)) < len(text) * 0.6:
            return translator.translate(text, dest='en').text
        return text
    except:
        return text


# Translate only top 200 comments to start with
sample_df = filtered_df.head(3000).copy()
sample_df['translated_comment'] = sample_df['comment'].apply(translate_if_needed)

filtered_df['translated_comment'] = filtered_df['comment'].apply(translate_if_needed)

import requests
import json
import time

PERSPECTIVE_API_KEY = "AIzaSyAv5BNAjJ2jNXHhd-haKxkCS2FlTChGL3k"  # Replace this
url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={PERSPECTIVE_API_KEY}"

def get_safety_scores(text):
    data = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {
            'TOXICITY': {}, 'INSULT': {}, 'PROFANITY': {},
            'THREAT': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}
        }
    }

    try:
        response = requests.post(url, data=json.dumps(data))
        result = response.json()

        if 'attributeScores' not in result:
            return {attr: None for attr in data['requestedAttributes']}

        return {
            attr: round(result['attributeScores'][attr]['summaryScore']['value'], 3)
            for attr in data['requestedAttributes']
        }

    except Exception as e:
        print("Error analyzing:", text)
        return {attr: None for attr in data['requestedAttributes']}

from tqdm import tqdm
tqdm.pandas()

scores = []
for comment in tqdm(sample_df['translated_comment'], desc="Scoring Comments"):
    score = get_safety_scores(comment)
    scores.append(score)
    time.sleep(1.2)

import pandas as pd

# Convert scores to DataFrame
scores_df = pd.DataFrame(scores)

# Combine with the original comments
final_df = pd.concat([filtered_df.reset_index(drop=True), scores_df], axis=1)

# Save it
final_df.to_csv("comments_with_safety_scores.csv", index=False)

# Preview
final_df.head()

final_df=pd.read_csv("/content/comments_with_safety_scores.csv")

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(final_df['TOXICITY'], bins=20, kde=True)
plt.title("Toxicity Score Distribution")
plt.xlabel("Toxicity Score")
plt.ylabel("Frequency")
plt.show()

test_comment = "You are so dumb and disgusting."

print(get_safety_scores(test_comment))

print(final_df.shape)
print(final_df.info())
print(final_df.describe())

# Check sample data
final_df.sample(5)

final_df['word_count'] = final_df['comment'].apply(lambda x: len(str(x).split()))
final_df['char_count'] = final_df['comment'].apply(lambda x: len(str(x)))

# Plot
plt.figure(figsize=(10, 5))
sns.histplot(final_df['word_count'], bins=50, kde=True)
plt.title("Word Count Distribution")
plt.xlabel("Words per Comment")
plt.show()

print(final_df.columns)

# You can adjust weights if needed — right now all are equally weighted
score_columns = ['TOXICITY', 'INSULT', 'PROFANITY', 'THREAT', 'SEVERE_TOXICITY']

# Fill missing values just in case
final_df[score_columns] = final_df[score_columns].fillna(0)

# Compute average safety index
final_df["safety_index"] = final_df[score_columns].mean(axis=1)

percentiles = final_df["safety_index"].quantile([0.90, 0.95, 0.99, 0.999])
print(percentiles)

def classify_safety_index(score):
    if score >= 0.3:       # Top 1% are truly unsafe
        return "unsafe"
    elif score >= 0.057:   # Top 5%–1% need review
        return "review"
    else:
        return "safe"

final_df["safety_label"] = final_df["safety_index"].apply(classify_safety_index)

sns.countplot(data=final_df, x="safety_label", palette="Set2")
plt.title("Safety Label Distribution (Based on Composite Index)")
plt.xlabel("Safety Category")
plt.ylabel("Count")
plt.show()

for label in ["unsafe", "review", "safe"]:
    print(f"\n🔹 {label.upper()} EXAMPLES:")
    print(final_df[final_df["safety_label"] == label]["comment"].dropna().sample(3).values)

plt.figure(figsize=(8, 6))
sns.heatmap(final_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

!pip install emoji

!pip install textblob textstat

import pandas as pd
import re
import string
import emoji
from textblob import TextBlob
import textstat

# Custom swear word list (expand as needed)
swear_words = [
    "damn", "hell", "idiot", "stupid", "hate", "kill", "shut up", "f***", "bitch", "moron"
]

# Function to extract features from final_df
def extract_text_features(final_df, text_col="comment"):
    final_df = final_df.copy()

    # Word & character counts
    final_df["word_count"] = final_df[text_col].apply(lambda x: len(str(x).split()))
    final_df["char_count"] = final_df[text_col].apply(lambda x: len(str(x)))

    # All CAPS ratio
    final_df["all_caps_ratio"] = final_df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )

    # Exclamation / Question marks
    final_df["exclamation_count"] = final_df[text_col].apply(lambda x: str(x).count("!"))
    final_df["question_count"] = final_df[text_col].apply(lambda x: str(x).count("?"))

    # Emoji count
    final_df["emoji_count"] = final_df[text_col].apply(lambda x: len([c for c in str(x) if c in emoji.EMOJI_DATA]))

    # Swear word count
    def count_swears(text):
        text = str(text).lower()
        return sum(word in text for word in swear_words)

    final_df["swear_word_count"] = final_df[text_col].apply(count_swears)

    # Sentiment polarity
    final_df["sentiment_polarity"] = final_df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Readability score
    final_df["readability_score"] = final_df[text_col].apply(lambda x: textstat.flesch_reading_ease(str(x)))

    # Repetition score (e.g., !!! or loooool)
    def repetition_score(text):
        text = str(text)
        repeated = re.findall(r'(.)\1{2,}', text)
        return len(repeated)

    final_df["repetition_score"] = final_df[text_col].apply(repetition_score)

    # Personal attack flag
    final_df["personal_attack_flag"] = final_df[text_col].apply(
        lambda x: int(bool(re.search(r'\byou\b.*\b(stupid|idiot|dumb|crazy|trash)\b', str(x).lower())))
    )

    # Contains a link
    final_df["has_link"] = final_df[text_col].str.contains(r"http|www", case=False, na=False).astype(int)

    return final_df

final_df = extract_text_features(final_df, text_col="comment")

# Print 5 sample rows with important features
print(final_df[[
    "comment", "word_count", "char_count", "swear_word_count",
    "emoji_count", "sentiment_polarity", "readability_score",
    "personal_attack_flag", "has_link"
]])

total_swears = final_df["swear_word_count"].sum()
total_chars = final_df["char_count"].sum()
total_emojis = final_df["emoji_count"].sum()
total_exclamations = final_df["exclamation_count"].sum()
total_questions = final_df["question_count"].sum()
total_repetitions = final_df["repetition_score"].sum()
total_links = final_df["has_link"].sum()
total_attacks = final_df["personal_attack_flag"].sum()

print("📊 TOTAL COUNTS ACROSS ALL COMMENTS:")
print(f"🧨 Swear words      : {total_swears}")
print(f"🔠 Characters       : {total_chars}")
print(f"😂 Emojis           : {total_emojis}")
print(f"❗ Exclamations     : {total_exclamations}")
print(f"❓ Question marks   : {total_questions}")
print(f"🔁 Repetitions      : {total_repetitions}")
print(f"🔗 Comments with links : {total_links}")
print(f"🗣️ Personal attacks : {total_attacks}")

final_df.to_csv("final_start_model.csv", index=False)

import pandas as pd

df2=pd.read_csv("/content/final_start_model.csv")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Define target
y = df2['safety_label']

# Define structured features
structured_features = [
    'swear_word_count', 'emoji_count', 'exclamation_count', 'question_count',
    'repetition_score', 'personal_attack_flag', 'has_link',
    'TOXICITY', 'INSULT', 'PROFANITY', 'THREAT', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK'
]

X_structured = df2[structured_features]

# Vectorize the text
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_text = tfidf.fit_transform(df2['comment'])

# Scale structured features
scaler = StandardScaler()
X_structured_scaled = scaler.fit_transform(X_structured)

# Combine TF-IDF and structured features
X_combined = hstack([X_text, X_structured_scaled])

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# Encode safety_label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df2["safety_label"])

# Define structured features (use no-leakage version if needed)
structured_features = [
    'swear_word_count', 'emoji_count', 'exclamation_count', 'question_count',
    'repetition_score', 'personal_attack_flag', 'has_link',
    'sentiment_polarity', 'readability_score'
]

X_structured = df2[structured_features]

# TF-IDF from comment
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_text = tfidf.fit_transform(df2["comment"])

# Scale structured features
scaler = StandardScaler()
X_structured_scaled = scaler.fit_transform(X_structured)

# Combine
X_combined = hstack([X_text, X_structured_scaled])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           cv=3,
                           scoring='f1_macro',
                           verbose=2,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

from sklearn.metrics import classification_report, confusion_matrix

# Predict
y_pred = best_model.predict(X_test)

# Decode back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Show metrics
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_test_labels, y_pred_labels))

print("✅ Best Hyperparameters Found:")
print(grid_search.best_params_)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Step 1: TF-IDF
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf.fit_transform(df2['comment'])

# Step 2: Convert to dense and scale
X_dense = X_tfidf.toarray()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dense)


import umap.umap_ as umap

umap_model = umap.UMAP(n_components=20, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_umap)
centroids = kmeans.cluster_centers_
assigned = centroids[clusters]


X_mean = X_umap.mean(axis=0)
TSS = np.sum((X_umap - X_mean) ** 2)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    preds = kmeans.fit_predict(X_umap)
    centroids = kmeans.cluster_centers_
    assigned = centroids[preds]
    BSS = np.sum((assigned - X_mean) ** 2)
    r2 = BSS / TSS
    print(f"Clusters: {k} → R²: {r2:.4f}")

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2),
    lowercase=True
)

from scipy.sparse import hstack

X_struct = df2[['emoji_count', 'swear_word_count']]  # example
X_combined = hstack([X_text, X_struct])

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np

# Refit final model with k=10
kmeans_final = KMeans(n_clusters=10, random_state=42)
clusters_final = kmeans_final.fit_predict(X_umap)
centroids_final = kmeans_final.cluster_centers_
assigned_centroids = centroids_final[clusters_final]

rmse_final = np.sqrt(mean_squared_error(X_umap, assigned_centroids))
print(f"📉 Final RMSE (KMeans, k=10): {rmse_final:.4f}")
X_mean = np.mean(X_umap, axis=0)
TSS = np.sum((X_umap - X_mean) ** 2)
BSS = np.sum((assigned_centroids - X_mean) ** 2)

r2_final = BSS / TSS
print(f"📈 Final R² (KMeans, k=10): {r2_final:.4f}")

y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_umap, clusters_final)
print(f"Silhouette Score (KMeans, k=10): {sil_score:.4f}")

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

sil_scores = []
k_values = range(2, 16)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    preds = kmeans.fit_predict(X_umap)  # make sure X_umap is 2D from UMAP
    score = silhouette_score(X_umap, preds)
    sil_scores.append(score)
    print(f"k={k} → Silhouette Score: {score:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(k_values, sil_scores, marker='o')
plt.title("Silhouette Score vs k (KMeans)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

!pip install --upgrade transformers datasets

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = df2[['comment', 'safety_label']].rename(columns={'comment': 'text', 'safety_label': 'label'})
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])   # e.g. safe→0, review→1, unsafe→2

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 5) Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )
dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format('torch', columns=['input_ids','attention_mask','label'])

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(le.classes_)
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'precision_macro': precision_score(labels, preds, average='macro'),
        'recall_macro': recall_score(labels, preds, average='macro')
    }

import transformers
print(transformers.__version__)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./distilbert-safety",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,

    # ★ log only to TensorBoard
    report_to=["tensorboard"],
    logging_dir="./runs",            # where tensorboard logs go
)

trainer.save_model("safety_distilbert")
tokenizer.save_pretrained("safety_distilbert")

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
model = DistilBertForSequenceClassification.from_pretrained("safety_distilbert")
tokenizer = DistilBertTokenizerFast.from_pretrained("safety_distilbert")