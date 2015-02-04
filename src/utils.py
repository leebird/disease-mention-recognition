from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import leveldb
from pycrfsuite import ItemSequence
from gensim.models import Word2Vec

# print('load level dbs and word2vec model')
atom_db = leveldb.LevelDB('data/atom_db')
bigram_db = leveldb.LevelDB('data/bigram_db')
# w2v_model = Word2Vec.load('/home/leebird/Projects/word2vec/word2vec/500-vec')

def word2features(sent, i):
    global atom_db, bigram_db
    # global w2v_model
    features = {}

    word = sent[i][0]
    postag = sent[i][1]

    try:
        atom_db.Get(word.lower().encode('utf-8'))
        in_db = True
    except KeyError:
        in_db = False

    # try:
    # vec = w2v_model[word.lower()]
    #     for j, ele in enumerate(vec):
    #         features['wordvec_'+str(j)] = ele
    # except KeyError:
    #     pass

    features.update({
        'bias': True,
        'word.lower': word.lower(),
        'word.in_db': in_db,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[+3:]': word[0:3],
        'word[+2:]': word[0:2],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.isfirst': (i == 0),
        'postag': postag,
        'postag[:2]': postag[:2],
    })

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]

        word1_word = word1.lower() + '|' + word.lower()

        try:
            bigram_db.Get(word1_word.encode('utf-8'))
            in_db = True
        except KeyError:
            in_db = False

        features.update({
            '-1:word.lower': word1.lower(),
            '-1:word.istitle': word1.istitle(),
            '-1:word.isupper': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],

            '<2:word.lower': word1.lower(),
            '<2:word.istitle': word1.istitle(),
            '<2:word.isupper': word1.isupper(),
            '<2:postag': postag1,
            '<2:postag[:2]': postag1[:2],

            'db_-1:word.lower|word.lower': in_db,
            '-1:word.lower|word.lower': word1_word,
            '-1:postag|postag': postag1 + '|' + postag,
            '-1:postag[:2]|postag[:2]': postag1[:2] + '|' + postag[:2],
        })
    else:
        features.update({'BOS': True})

    if i > 1:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        word2 = sent[i - 2][0]
        postag2 = sent[i - 2][1]
        word2_word1 = word2.lower() + '|' + word1.lower()

        try:
            bigram_db.Get(word2_word1.encode('utf-8'))
            in_db = True
        except KeyError:
            in_db = False

        features.update({
            '-2:word.lower': word2.lower(),
            '-2:word.istitle': word2.istitle(),
            '-2:word.isupper': word2.isupper(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],

            '<2:word.lower': word2.lower(),
            '<2:word.istitle': word2.istitle(),
            '<2:word.isupper': word2.isupper(),
            '<2:postag': postag2,
            '<2:postag[:2]': postag2[:2],

            'db_-2:word.lower|-1:word.lower': in_db,
            '-2:word.lower|-1:word.lower': word2_word1,
            '-2:postag|-1:postag': postag2 + '|' + postag1,
            '-2:postag[:2]|-1:postag[:2]': postag2[:2] + '|' + postag1[:2],
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]

        word_word1 = word.lower() + '|' + word1.lower()

        try:
            bigram_db.Get(word_word1.encode('utf-8'))
            in_db = True
        except KeyError:
            in_db = False

        features.update({
            '+1:word.lower': word1.lower(),
            '+1:word.istitle': word1.istitle(),
            '+1:word.isupper': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],

            '>2:word.lower': word1.lower(),
            '>2:word.istitle': word1.istitle(),
            '>2:word.isupper': word1.isupper(),
            '>2:postag': postag1,
            '>2:postag[:2]': postag1[:2],

            'db_word.lower|+1:word.lower': in_db,
            'word.lower|+1:word.lower': word_word1,
            'postag|+1:postag': postag + '|' + postag1,
            'postag[:2]|+1:postag[:2]': postag[:2] + '|' + postag1[:2],
        })
    else:
        features.update({'EOS': True})

    if i < len(sent) - 2:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]

        word1_word2 = word1.lower() + '|' + word2.lower()

        try:
            bigram_db.Get(word1_word2.encode('utf-8'))
            in_db = True
        except KeyError:
            in_db = False

        features.update({
            '+2:word.lower': word2.lower(),
            '+2:word.istitle': word2.istitle(),
            '+2:word.isupper': word2.isupper(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],

            '>2:word.lower': word2.lower(),
            '>2:word.istitle': word2.istitle(),
            '>2:word.isupper': word2.isupper(),
            '>2:postag': postag2,
            '>2:postag[:2]': postag2[:2],

            'db_+1:word.lower|+2:word.lower': in_db,
            '+1:word.lower|+2:word.lower': word1_word2,
            '+1:postag|+2:postag': postag1 + '|' + postag2,
            '+1:postag[:2]|+2:postag[:2]': postag1[:2] + '|' + postag2[:2],
        })

    # for j in range(i - 1, -1, -1):
    # if sent[j][1].startswith('N'):
    # features.append('-1:noun.lower=' + sent[j][0].lower())
    # break
    #
    # for j in range(i, len(sent)):
    # if sent[j][1].startswith('N'):
    # features.append('+1:noun.lower=' + sent[j][0].lower())
    #         break

    return features


def sent2features(sent):
    return ItemSequence([word2features(sent, i) for i in range(len(sent))])


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def get_sentences(sentence_file):
    with open(sentence_file, 'r') as handler:
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
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )