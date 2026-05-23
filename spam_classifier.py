 # ============================================================
# SMS SPAM DETECTION SYSTEM
# Arch Technologies - Machine Learning Internship
# ============================================================

# --- loading all necessary libraries ---
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: READING THE DATASET
# ============================================================

def read_dataset():
    """
    Read the SMS spam dataset from local CSV file.
    The CSV contains two main columns: category (spam/ham) and the text message.
    If local file is not available, it downloads from a public URL as backup.
    """
    file_path = os.path.join('data', 'spam.csv')

    if os.path.exists(file_path):
        print("[+] Reading dataset from local CSV file...")
        data = pd.read_csv(file_path, encoding='latin-1')

        # drop extra unnamed columns that come with kaggle download
        data = data[['v1', 'v2']]
        data.columns = ['category', 'text']
    else:
        print("[+] Local file not found, fetching from online source...")
        link = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        data = pd.read_csv(link, sep='\t', header=None, names=['category', 'text'])

    print(f"[+] Dataset loaded! Total records: {data.shape[0]}, Columns: {data.shape[1]}")
    return data


# ============================================================
# PART 2: DATA EXPLORATION
# ============================================================

def analyze_dataset(data):
    """
    Perform initial analysis on the dataset to understand its structure.
    Check for missing values, class distribution, and message characteristics.
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)

    # dataset overview
    print(f"\nTotal records: {len(data)}")
    print(f"Column names: {list(data.columns)}")
    print(f"\nMissing values:\n{data.isnull().sum()}")

    # check duplicates
    dup_count = data.duplicated().sum()
    print(f"\nDuplicate entries found: {dup_count}")

    # class breakdown
    print(f"\nCategory Breakdown:")
    category_dist = data['category'].value_counts()
    for cat, num in category_dist.items():
        percentage = num / len(data) * 100
        print(f"  {cat}: {num} messages ({percentage:.1f}%)")

    # add length features for analysis
    data['char_count'] = data['text'].apply(len)
    data['total_words'] = data['text'].apply(lambda x: len(x.split()))

    print(f"\nAverage Character Count per Category:")
    print(data.groupby('category')['char_count'].mean().to_string())

    print(f"\nAverage Word Count per Category:")
    print(data.groupby('category')['total_words'].mean().to_string())

    # display some example messages
    print(f"\n--- Example HAM (legitimate) messages ---")
    ham_examples = data[data['category'] == 'ham']['text'].head(3)
    for idx, msg in enumerate(ham_examples, 1):
        print(f"  {idx}. {msg[:80]}...")

    print(f"\n--- Example SPAM messages ---")
    spam_examples = data[data['category'] == 'spam']['text'].head(3)
    for idx, msg in enumerate(spam_examples, 1):
        print(f"  {idx}. {msg[:80]}...")

    return data


def generate_eda_charts(data):
    """
    Generate exploratory data analysis charts:
    - Category count bar chart
    - Category percentage pie chart
    - Character length histograms for ham and spam messages
    """
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    chart_colors = ['#2ecc71', '#e67e22']

    # chart 1: bar chart of category counts
    data['category'].value_counts().plot(kind='bar', ax=ax[0, 0], color=chart_colors)
    ax[0, 0].set_title('Message Count by Category', fontsize=13)
    ax[0, 0].set_xlabel('Category')
    ax[0, 0].set_ylabel('Number of Messages')
    ax[0, 0].tick_params(axis='x', rotation=0)

    # chart 2: pie chart
    data['category'].value_counts().plot(kind='pie', ax=ax[0, 1],
                                          autopct='%1.1f%%', colors=chart_colors,
                                          startangle=90)
    ax[0, 1].set_title('Category Percentage Split', fontsize=13)
    ax[0, 1].set_ylabel('')

    # chart 3: ham message length distribution
    data[data['category'] == 'ham']['char_count'].hist(bins=50, ax=ax[1, 0],
                                                        color='#2ecc71', alpha=0.7)
    ax[1, 0].set_title('Ham Messages - Character Length Spread', fontsize=13)
    ax[1, 0].set_xlabel('Number of Characters')
    ax[1, 0].set_ylabel('Frequency')

    # chart 4: spam message length distribution
    data[data['category'] == 'spam']['char_count'].hist(bins=50, ax=ax[1, 1],
                                                         color='#e67e22', alpha=0.7)
    ax[1, 1].set_title('Spam Messages - Character Length Spread', fontsize=13)
    ax[1, 1].set_xlabel('Number of Characters')
    ax[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('exploration_charts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[+] EDA charts saved as 'exploration_charts.png'")


# ============================================================
# PART 3: TEXT CLEANING
# ============================================================

def clean_message(raw_text):
    """
    Clean a single text message through these steps:
    1. Convert everything to lowercase for uniformity
    2. Strip out special characters, numbers, and punctuation
    3. Remove extra whitespace
    4. Filter out common English stopwords (a, the, is, etc.)
    5. Apply Porter Stemming to reduce words to base form
       Example: 'winning' -> 'win', 'played' -> 'play'
    """
    ps = PorterStemmer()
    english_stops = set(stopwords.words('english'))

    # make everything lowercase
    raw_text = raw_text.lower()

    # keep only alphabetic characters
    raw_text = re.sub(r'[^a-zA-Z\s]', '', raw_text)

    # clean up whitespace
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()

    # break into individual words
    tokens = raw_text.split()

    # remove stopwords and stem the remaining words
    # ignore very short words (2 chars or less) as they are usually noise
    filtered = []
    for token in tokens:
        if token not in english_stops and len(token) > 2:
            filtered.append(ps.stem(token))

    return ' '.join(filtered)


def run_preprocessing(data):
    """
    Apply text cleaning to every message in the dataset.
    Also convert category labels to binary values: ham=0, spam=1.
    """
    print("\n[+] Starting text preprocessing...")

    data['processed_text'] = data['text'].apply(clean_message)

    # show before and after examples
    print("\nCleaning Examples (Before -> After):")
    for i in range(3):
        print(f"\n  Original:  {data['text'].iloc[i][:70]}...")
        print(f"  Cleaned:   {data['processed_text'].iloc[i][:70]}...")

    # encode labels as numbers
    data['is_spam'] = data['category'].map({'ham': 0, 'spam': 1})

    print(f"\n[+] Preprocessing done!")
    print(f"  Label encoding: ham = 0, spam = 1")
    return data


# ============================================================
# PART 4: CONVERTING TEXT TO NUMBERS (TF-IDF)
# ============================================================

def vectorize_text(data):
    """
    Transform text messages into numerical feature vectors using TF-IDF.

    Why TF-IDF?
    - Machine learning models cannot work with raw text directly
    - TF-IDF converts each message into a vector of numbers
    - It measures how important each word is to a specific message
    - Words that appear often in one message but rarely across all messages
      get higher importance scores
    - max_features=3000 limits to the top 3000 most relevant words
    """
    print("\n[+] Converting text to numerical features using TF-IDF...")

    vectorizer = TfidfVectorizer(max_features=3000)

    features = vectorizer.fit_transform(data['processed_text'])
    labels = data['is_spam']

    print(f"  Feature matrix shape: {features.shape[0]} messages x {features.shape[1]} word features")
    print(f"  Using top {features.shape[1]} words as features")

    # display some of the feature words
    word_features = vectorizer.get_feature_names_out()
    print(f"\n  Sample feature words: {list(word_features[:10])}")

    return features, labels, vectorizer


# ============================================================
# PART 5: SPLITTING DATA
# ============================================================

def create_train_test_sets(features, labels):
    """
    Divide the dataset into training (80%) and testing (20%) portions.
    - stratify ensures both sets maintain the same spam/ham ratio
    - random_state ensures the split is reproducible every time
    """
    feat_train, feat_test, lab_train, lab_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\n[+] Data splitting done:")
    print(f"  Training samples: {feat_train.shape[0]}")
    print(f"  Testing samples:  {feat_test.shape[0]}")

    return feat_train, feat_test, lab_train, lab_test


# ============================================================
# PART 6: TRAINING CLASSIFIERS
# ============================================================

def build_naive_bayes(feat_train, lab_train):
    """
    Train a Multinomial Naive Bayes classifier.
    This algorithm works particularly well for text classification tasks
    because it handles word frequency data efficiently.
    It is fast to train and gives strong baseline results.
    """
    print("\n[+] Training Naive Bayes classifier...")
    nb_clf = MultinomialNB(alpha=1.0)
    nb_clf.fit(feat_train, lab_train)
    print("  Naive Bayes training finished!")
    return nb_clf


def build_logistic_regression(feat_train, lab_train):
    """
    Train a Logistic Regression classifier as a second model.
    Comparing two different algorithms helps us pick the better one
    for this particular dataset.
    """
    print("[+] Training Logistic Regression classifier...")
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(feat_train, lab_train)
    print("  Logistic Regression training finished!")
    return lr_clf


# ============================================================
# PART 7: MEASURING PERFORMANCE
# ============================================================

def check_performance(model, name, feat_test, lab_test):
    """
    Measure how well a model performs on unseen test data.
    Metrics explained:
    - Accuracy: percentage of all predictions that were correct
    - Precision: of all messages flagged as spam, how many truly were spam
    - Recall: of all actual spam messages, how many did the model catch
    - F1 Score: balanced combination of precision and recall
    """
    preds = model.predict(feat_test)

    acc_val = accuracy_score(lab_test, preds)
    prec_val = precision_score(lab_test, preds)
    rec_val = recall_score(lab_test, preds)
    f1_val = f1_score(lab_test, preds)

    print(f"\n{'='*50}")
    print(f"  {name} - Performance Report")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc_val:.4f} ({acc_val*100:.1f}%)")
    print(f"  Precision: {prec_val:.4f}")
    print(f"  Recall:    {rec_val:.4f}")
    print(f"  F1 Score:  {f1_val:.4f}")
    print(f"\n  Full Classification Report:")
    print(classification_report(lab_test, preds,
                                target_names=['Legitimate', 'Spam']))

    return preds, acc_val


def draw_confusion_charts(lab_test, nb_preds, lr_preds, nb_acc, lr_acc):
    """
    Draw confusion matrices for both models.
    The confusion matrix breaks down predictions into four categories:
    - True Negative: correctly identified as ham
    - True Positive: correctly identified as spam
    - False Positive: ham message wrongly flagged as spam
    - False Negative: spam message that slipped through as ham
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # naive bayes matrix
    nb_cm = confusion_matrix(lab_test, nb_preds)
    sns.heatmap(nb_cm, annot=True, fmt='d', cmap='YlGn', ax=ax[0],
                xticklabels=['Legitimate', 'Spam'], yticklabels=['Legitimate', 'Spam'])
    ax[0].set_title(f'Naive Bayes (Acc: {nb_acc:.2%})', fontsize=13)
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')

    # logistic regression matrix
    lr_cm = confusion_matrix(lab_test, lr_preds)
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='PuBu', ax=ax[1],
                xticklabels=['Legitimate', 'Spam'], yticklabels=['Legitimate', 'Spam'])
    ax[1].set_title(f'Logistic Regression (Acc: {lr_acc:.2%})', fontsize=13)
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('evaluation_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[+] Confusion matrices saved as 'evaluation_matrices.png'")


def draw_accuracy_comparison(nb_acc, lr_acc):
    """
    Create a side-by-side bar chart comparing accuracy of both models.
    """
    model_names = ['Naive Bayes', 'Logistic Regression']
    acc_scores = [nb_acc, lr_acc]
    bar_colors = ['#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, acc_scores, color=bar_colors, width=0.5)

    # display accuracy value on top of each bar
    for bar, score in zip(bars, acc_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.2%}', ha='center', fontsize=13, fontweight='bold')

    ax.set_ylim(0.9, 1.0)
    ax.set_title('Accuracy Comparison Between Models', fontsize=14)
    ax.set_ylabel('Accuracy Score')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[+] Accuracy comparison chart saved as 'accuracy_comparison.png'")


# ============================================================
# PART 8: TESTING ON NEW MESSAGES
# ============================================================

def predict_new_messages(model, vectorizer):
    """
    Run the trained model on custom messages that it has never seen before.
    This demonstrates the model's ability to generalize to real-world inputs.
    """
    sample_messages = [
        "You have been selected as a winner! Claim your FREE prize now!",
        "Hey, can we reschedule our meeting to 3pm?",
        "ALERT: Suspicious activity detected on your account. Verify immediately.",
        "Please bring your laptop to class tomorrow.",
        "Earn thousands weekly!!! Reply YES to start winning prizes now!",
        "Dad is asking if you want anything from the store",
        "Click HERE for exclusive deals! Limited time offer expires today!",
        "See you at the coffee shop around 5, bring the notes"
    ]

    print(f"\n{'='*60}")
    print("  PREDICTIONS ON NEW MESSAGES")
    print(f"{'='*60}")

    for msg in sample_messages:
        cleaned = clean_message(msg)
        transformed = vectorizer.transform([cleaned])
        result = model.predict(transformed)[0]
        tag = "SPAM" if result == 1 else "LEGITIMATE"
        marker = "[!!]" if result == 1 else "[ok]"

        print(f"\n  {marker} [{tag}]")
        print(f"       \"{msg[:65]}{'...' if len(msg) > 65 else ''}\"")


# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================

def main():
    print("=" * 60)
    print("  SMS SPAM DETECTION SYSTEM")
    print("  Arch Technologies - ML Internship")
    print("=" * 60)

    # part 1: read the data
    data = read_dataset()

    # part 2: explore and visualize
    data = analyze_dataset(data)
    generate_eda_charts(data)

    # part 3: clean the text
    data = run_preprocessing(data)

    # part 4: convert text to numbers
    features, labels, vectorizer = vectorize_text(data)

    # part 5: split into train and test
    feat_train, feat_test, lab_train, lab_test = create_train_test_sets(features, labels)

    # part 6: train both models
    nb_clf = build_naive_bayes(feat_train, lab_train)
    lr_clf = build_logistic_regression(feat_train, lab_train)

    # part 7: measure performance
    nb_preds, nb_acc = check_performance(nb_clf, "Naive Bayes", feat_test, lab_test)
    lr_preds, lr_acc = check_performance(lr_clf, "Logistic Regression", feat_test, lab_test)

    # generate evaluation plots
    draw_confusion_charts(lab_test, nb_preds, lr_preds, nb_acc, lr_acc)
    draw_accuracy_comparison(nb_acc, lr_acc)

    # part 8: test with custom messages
    winner = nb_clf if nb_acc >= lr_acc else lr_clf
    winner_name = "Naive Bayes" if nb_acc >= lr_acc else "Logistic Regression"
    print(f"\n[+] Best performing model: {winner_name}")

    predict_new_messages(winner, vectorizer)

    # display final summary
    print(f"\n{'='*60}")
    print("  PROJECT SUMMARY")
    print(f"{'='*60}")
    print(f"  Naive Bayes Accuracy:         {nb_acc:.2%}")
    print(f"  Logistic Regression Accuracy: {lr_acc:.2%}")
    print(f"  Selected Model: {winner_name}")
    print(f"\n  Output Files Created:")
    print(f"    - exploration_charts.png")
    print(f"    - evaluation_matrices.png")
    print(f"    - accuracy_comparison.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
