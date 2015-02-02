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


train_sents = list(get_sentences('corpus/BIO/train.bio'))

# print(sent2features(train_sents[0])[0])

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,  # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('ncbi_disease_train.model')

