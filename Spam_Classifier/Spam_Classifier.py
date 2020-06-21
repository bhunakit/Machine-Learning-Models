import numpy as np
import os
import tarfile
import urllib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import *
from collections import Counter
import email
import email.policy
import pickle
from Preprocessing import *


DOWNLOAD_ROOT = 'http://spamassassin.apache.org/old/publiccorpus/'
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join('datasets', 'spam')

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()

fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, 'easy_ham')
SPAM_DIR = os.path.join(SPAM_PATH, 'spam')
ham_fnames = [name for name in os.listdir(HAM_DIR) if len(name) > 20]
spam_fnames = [name for name in os.listdir(SPAM_DIR) if len(name) > 20]


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_fnames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_fnames]

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


x = np.array(ham_emails + spam_emails)
y = np.array([0]*len(ham_emails) + [1]*len(spam_emails))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)


pipeline = Pipeline([
    ('email_to_count', email_to_counter()),
    ('count_to_vector', counter_to_vector())
])

x_train_transformed = pipeline.fit_transform(x_train)

log_clf = LogisticRegression(solver="liblinear", random_state=42)

x_test_transformed = pipeline.transform(x_test)

log_clf.fit(x_train_transformed, y_train)

y_pred = log_clf.predict(x_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))


pickle.dump(log_clf, open('spam_filter', 'wb'))

