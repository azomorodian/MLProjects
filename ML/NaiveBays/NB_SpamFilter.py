import pandas as pd
import numpy as np
from collections import defaultdict
import re
df = pd.read_csv('../input/SMS_SpamCollection/SMSSpamCollection',
                 sep='\t',
                 header=None,
                 names=['Label', 'SMS'])
sms_spam = df.copy()
print(sms_spam.head())
#sms_spam.columns = ['Label', 'SMS']
#sms_spam.columns = ['label', 'message']
print(sms_spam['Label'].value_counts())
print(f"Total messages: {len(sms_spam)}")
data_randomized = sms_spam.sample(frac=1, random_state=1)
training_test_index = round(len(data_randomized) * 0.8)
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)
print(f"Training set size: {training_set.shape}")
print(f"Test set size: {test_set.shape}")
def clean_text(text):
    """
    Clean text by converting to lowercase, removing punctuation,
    and splitting into words
    """
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Split into words
    words = text.split()
    return words
# Create a cleaned version of the training set
training_set_clean = training_set.copy()
training_set_clean['SMS'] = training_set_clean['SMS'].apply(clean_text)
print("Original message:", training_set['SMS'])
print("Cleaned message:", training_set_clean['SMS'])

vocabulary = set()
for message in training_set_clean['SMS']:
    vocabulary.update(message)
vocabulary = sorted(list(vocabulary))
print(f"Vocabulary size: {len(vocabulary)}")
print(f"First 50 words: {vocabulary[:50]}")

# Create word frequency tables for spam and ham
word_frequencies_spam = defaultdict(int)
word_frequencies_ham = defaultdict(int)
# Iterate through training set
for idx, row in training_set_clean.iterrows():
    if row['Label'] == 'spam':
        for word in row['SMS']:
            word_frequencies_spam[word] += 1
    else:
        for word in row['SMS']:
            word_frequencies_ham[word] += 1

# Calculate prior probabilities
total_spam = (training_set['Label'] == 'spam').sum()
total_ham = (training_set['Label'] == 'ham').sum()
p_spam = total_spam / len(training_set)
p_ham = total_ham / len(training_set)
print(f"P(Spam) = {p_spam:.4f}")
print(f"P(Ham) = {p_ham:.4f}")

# Total words in spam and ham categories
total_words_spam = sum(word_frequencies_spam.values())
total_words_ham = sum(word_frequencies_ham.values())
print(f"Total words in spam messages: {total_words_spam}")
print(f"Total words in ham messages: {total_words_ham}")

#Handling the Zero Probability Problem
# Laplace smoothing parameter
alpha = 1
# Calculate conditional probabilities P(word|spam) and P(word|ham)
def calculate_conditional_prob(word, label_type, alpha):
    """Calculate P(word|label)"""
    if label_type == 'spam':
        word_count = word_frequencies_spam.get(word, 0)
        total_words = total_words_spam
    else:
        word_count = word_frequencies_ham.get(word, 0)
        total_words = total_words_ham
    return (word_count + alpha) / (total_words + alpha * len(vocabulary))
# Test it
test_word = "winner"
p_word_spam = calculate_conditional_prob(test_word, 'spam', alpha)
p_word_ham = calculate_conditional_prob(test_word, 'ham', alpha)
print(f"P('{test_word}'|Spam) = {p_word_spam:.6f}")
print(f"P('{test_word}'|Ham) = {p_word_ham:.6f}")


def classify_message(message, alpha):
    """Classify a message as spam or ham"""
    # Clean the message
    words = clean_text(message)

    # Calculate log probabilities to avoid underflow
    log_p_spam = np.log(p_spam)
    log_p_ham = np.log(p_ham)

    # Multiply probabilities for each word
    for word in words:
        if word in vocabulary:
            log_p_spam += np.log(calculate_conditional_prob(word, 'spam', alpha))
            log_p_ham += np.log(calculate_conditional_prob(word, 'ham', alpha))
    # Convert back from log space
    p_spam_given_message = np.exp(log_p_spam)
    p_ham_given_message = np.exp(log_p_ham)
    # Return classification
    if p_spam_given_message > p_ham_given_message:
        return 'spam', p_spam_given_message / (p_spam_given_message + p_ham_given_message)
    else:
        return 'ham', p_ham_given_message / (p_spam_given_message + p_ham_given_message)
# Test on some examples
test_messages = [
    "WINNER!! This is the secret code to unlock the money: C3421.",
    "Hey, how are you doing today? Let's catch up soon!",
    "Congratulations! You've won a free iPhone! Click here now!"
]
for msg in test_messages:
    classification, confidence = classify_message(msg, alpha)
    print(f"Message: {msg[:50]}...")
    print(f"Classification: {classification.upper()} (Confidence: {confidence:.2%})\n")

