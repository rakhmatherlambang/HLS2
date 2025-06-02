import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Buat folder output jika belum ada
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load data
train_df = pd.read_csv('data/split/train.csv')
val_df = pd.read_csv('data/split/val.csv')
test_df = pd.read_csv('data/split/test.csv')

# Gabungkan train + val
X_train = pd.concat([train_df['processed_text'], val_df['processed_text']])
y_train = pd.concat([train_df['SDG_Category'], val_df['SDG_Category']])

X_test = test_df['processed_text']
y_test = test_df['SDG_Category']

# Pipeline model
vectorizer = TfidfVectorizer()
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('nb', MultinomialNB()),
    ('svc', SVC(kernel='linear', probability=True))
], voting='soft')

pipeline = make_pipeline(vectorizer, voting_clf)

# Training
print("🔧 Training model...")
pipeline.fit(X_train, y_train)

# Evaluasi
print("📊 Evaluating on test set...")
y_pred = pipeline.predict(X_test)

# Save classification report
report = classification_report(y_test, y_pred)
with open('reports/classification_report.txt', 'w') as f:
    f.write(report)
print("✅ Classification report saved to reports/classification_report.txt")

# Save confusion matrix
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels,
            yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved to reports/confusion_matrix.png")

# Save model
joblib.dump(pipeline, 'models/voting_classifier.pkl')
print("💾 Model saved to models/voting_classifier.pkl")
