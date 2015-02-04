import sys
import os

root = os.path.dirname(os.path.dirname(__file__))
legonlp = os.path.join(root, 'data/legonlp-master')
sys.path.append(legonlp)

import nltk
import nltk.data
import sys
from nltk.tokenize import WordPunctTokenizer
nltk.data.path = ['data']

import pycrfsuite
from .utils import sent2features, sent2labels, bio_classification_report, get_sentences, sent2tokens
from .bio_to_ann import bio_to_ann

def make_lines(tokens, pred):
    for tokens, labels in zip(tokens, pred):
        for token, label in zip(tokens, labels):
            yield token + '\t' + label + '\n'
        yield '\n'

def recognize(filepath):
    global sent_detector, word_punct_tokenizer
    
    handler = open(filepath, 'r')
    text = handler.read()
    handler.close()

    sentences = sent_detector.tokenize(text.strip())

    test_sents = []
    for sentence in sentences:
        # tokenization
        tokens = word_punct_tokenizer.tokenize(sentence)
        # get pos tag
        pos_tags = nltk.pos_tag(tokens)
        test_sents.append(pos_tags)
        
    featuers = [sent2features(s) for s in test_sents]
    tokens = [[pos_token[0] for pos_token in sent] for sent in test_sents]
    predictions = [tagger.tag(xseq) for xseq in featuers]

    bio_lines = make_lines(tokens, predictions)
    annotation = bio_to_ann(bio_lines, 'Disease')
    #TODO: map back to original text
    print(annotation)
    
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('specify text file/folder and model name')
        sys.exit(0)

    text_path = sys.argv[1]
    model_file = sys.argv[2]

    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)

    # sentence splitter
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # use WordPunctTokenizer to split disease-suppressor to disease, -, suppressor
    word_punct_tokenizer = WordPunctTokenizer()

    if os.path.isdir(text_path):
        for root,_,files in text_path:
            for f in files:
                if not f.endswith('.txt'):
                    continue
                path = os.path.join(root, f)
    elif os.path.isfile(text_path):
        recognize(text_path)




