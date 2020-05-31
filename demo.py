from nbsvm import NBSVM
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Get dataset
cats = ['alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'talk.politics.misc',
        'rec.sport.hockey']

newsgroups = fetch_20newsgroups(categories=cats)
x = TfidfVectorizer(ngram_range=(1,3)).fit_transform(newsgroups.data)
y = newsgroups.target
y = np.array(cats)[y]
print("Labels:", y)

# Split dataset
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2)
print(f"x_train.shape={x_train.shape}, y_train.shape:{y_train.shape}, \
x_test.shape:{x_test.shape}, y_test.shape:{y_test.shape}")

# Train
clf = NBSVM(n_jobs=-1)
clf.fit(x_train, y_train)

# Evaluate
print("Test Acc:", clf.evaluate(x_test, y_test))

# Predict Probabilites
print(f"-\n[Current Model] Predict Probabilites for first 3 sample:\n{clf.predict_proba(x[:3])}")

# Dump and restore model
import joblib
from pathlib import Path
dumped_model = Path("tmp.model.joblib")
joblib.dump(clf, dumped_model)
clf2 = joblib.load(dumped_model)
print(f"-\n[Restored Model] Predict Probabilites for first 3 sample:\n{clf2.predict_proba(x[:3])}")
print(f"Dumped Model size: {dumped_model.stat().st_size/1024**2:.1f} MiB")
dumped_model.unlink()
print("Remove dumped model.")