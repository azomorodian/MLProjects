from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

# خواندن داده‌ها
df = pd.read_csv('../input/SMS_SpamCollection/SMSSpamCollection',
                 sep='\t',
                 header=None,
                 names=['Label', 'SMS'])

# تصادفی کردن داده‌ها و تقسیم به آموزش و تست
data_randomized = df.sample(frac=1, random_state=1)
training_test_index = round(len(data_randomized) * 0.8)
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

# آماده‌سازی X و y
X_train = training_set['SMS']
y_train = training_set['Label']
X_test = test_set['SMS']
y_test = test_set['Label']

# تبدیل متن به وکتور (CountVectorizer)
vectorizer = CountVectorizer(lowercase=True)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_vectors.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# آموزش مدل
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_train_vectors, y_train)

# پیش‌بینی روی داده‌های تست
y_pred = classifier.predict(X_test_vectors)

# ارزیابی مدل
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# اصلاح نمایش ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# استخراج مقادیر TN, FP, FN, TP
# چون کلاس‌ها به ترتیب حروف الفبا هستند: 0=ham, 1=spam
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives (Ham correctly identified): {tn}")
print(f"False Positives (Ham incorrectly marked as Spam): {fp}")
print(f"False Negatives (Spam missed): {fn}")
print(f"True Positives (Spam correctly identified): {tp}")


# --- تابع اصلاح شده برای پیش‌بینی پیام جدید ---
def classify_new_message(message):
    """Use the trained model to classify a new message"""
    message_vector = vectorizer.transform([message])

    # گرفتن خودِ مقدار پیش‌بینی (رشته) به جای آرایه
    prediction = classifier.predict(message_vector)[0]

    # گرفتن احتمالات
    probabilities = classifier.predict_proba(message_vector)[0]

    # پیدا کردن ایندکس مربوط به کلاس 'spam'
    # classifier.classes_ معمولاً ['ham', 'spam'] است
    spam_index = list(classifier.classes_).index('spam')

    # استخراج احتمال اسپم بودن
    spam_prob = probabilities[spam_index]

    return prediction, spam_prob


# تست تابع
new_messages = [
    "Your package has been delivered",
    "Act now! Limited time offer! FREE CASH!!!",
    "Can we schedule a meeting tomorrow?",
    "Click here to claim your prize - you've won!"
]

print("\nTesting the classifier on new messages:\n")
for msg in new_messages:
    pred, prob = classify_new_message(msg)
    print(f"Message: {msg}")
    # چون pred الان رشته است، upper روی آن کار می‌کند
    print(f"Prediction: {pred.upper()} (Spam probability: {prob:.2%})\n")