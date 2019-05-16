from nbsvm import NBSVM
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

cats = ['alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'talk.politics.misc',
        'rec.sport.hockey']

newsgroups = fetch_20newsgroups(categories=cats)
x = TfidfVectorizer(ngram_range=(1,3)).fit_transform(newsgroups.data)
print("Lables: ", newsgroups.target)

x_train, x_test, y_train, y_test =  train_test_split(
    x, newsgroups.target,
    test_size=0.2)

print("x_train.shape, y_train.shape, x_test.shape, y_test.shape", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

clf = NBSVM(class_num=len(cats))
clf.fit(x_train, y_train)
print("Test Acc:", clf.evaluate(x_test, y_test))

print("Predict Probabilites for first 3 sample:\n", clf.predict_prob(x[:3]))