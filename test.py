from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

def get_sentences(sentence_file):
    handler = open(sentence_file, 'r')

    sentence = []
    for line in handler:
        line = line.strip()
        if len(line) == 0:
            yield sentence
            sentence = []
        else:
            tokens = tuple(line.split('\t'))
            sentence.append(tokens)

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
        )

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.start.isupper=%s' % word[0].isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        ]

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],

            '<2:word.lower=' + word1.lower(),
            '<2:word.istitle=%s' % word1.istitle(),
            '<2:word.isupper=%s' % word1.isupper(),
            '<2:postag=' + postag1,
            '<2:postag[:2]=' + postag1[:2],
            ])
    else:
        features.append('BOS')

    if i > 1:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        word2 = sent[i - 2][0]
        postag2 = sent[i - 2][1]
        features.extend([
            '-2:word.lower=' + word2.lower(),
            '-2:word.istitle=%s' % word2.istitle(),
            '-2:word.isupper=%s' % word2.isupper(),
            '-2:postag=' + postag2,
            '-2:postag[:2]=' + postag2[:2],

            '<2:word.lower=' + word1.lower(),
            '<2:word.istitle=%s' % word1.istitle(),
            '<2:word.isupper=%s' % word1.isupper(),
            '<2:postag=' + postag1,
            '<2:postag[:2]=' + postag1[:2],

            '<2:word.lower=' + word2.lower(),
            '<2:word.istitle=%s' % word2.istitle(),
            '<2:word.isupper=%s' % word2.isupper(),
            '<2:postag=' + postag2,
            '<2:postag[:2]=' + postag2[:2],
            ])


    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],

            '>2:word.lower=' + word1.lower(),
            '>2:word.istitle=%s' % word1.istitle(),
            '>2:word.isupper=%s' % word1.isupper(),
            '>2:postag=' + postag1,
            '>2:postag[:2]=' + postag1[:2],
            ])

    else:
        features.append('EOS')

    if i < len(sent) - 2:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]
        features.extend([
            '+2:word.lower=' + word2.lower(),
            '+2:word.istitle=%s' % word2.istitle(),
            '+2:word.isupper=%s' % word2.isupper(),
            '+2:postag=' + postag2,
            '+2:postag[:2]=' + postag2[:2],

            '>2:word.lower=' + word1.lower(),
            '>2:word.istitle=%s' % word1.istitle(),
            '>2:word.isupper=%s' % word1.isupper(),
            '>2:postag=' + postag1,
            '>2:postag[:2]=' + postag1[:2],

            '>2:word.lower=' + word2.lower(),
            '>2:word.istitle=%s' % word2.istitle(),
            '>2:word.isupper=%s' % word2.isupper(),
            '>2:postag=' + postag2,
            '>2:postag[:2]=' + postag2[:2],
            ])

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

tagger = pycrfsuite.Tagger()
tagger.open('ncbi_disease_train.model')

dev_sents = list(get_sentences('corpus/BIO/development.bio'))
X_test = [sent2features(s) for s in dev_sents]
y_test = [sent2labels(s) for s in dev_sents]

y_pred = [tagger.tag(xseq) for xseq in X_test]
print(bio_classification_report(y_test, y_pred))

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])