import sys
import os

root = os.path.dirname(os.path.dirname(__file__))
legonlp = os.path.join(root, 'data/legonlp-master')
sys.path.append(legonlp)

from annotation.writers import AnnWriter
from annotation.annotation import Annotation

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


def map_to_original(annotation, original_text):
    # TODO: integrate into Annotation
    text = annotation.text.strip()
    tokens = annotation.text.strip().split(' ')
    tokens = [token.strip() for token in tokens if len(token.strip()) > 0]
    entity_start_hash = {}
    entity_end_hash = {}
    for entity in annotation.entities:
        entity_start_hash[entity.start] = entity
        entity_end_hash[entity.end] = entity
    index = 0
    index_ori = 0

    for token in tokens:

        index = text.find(token, index)
        index_ori = original_text.find(token, index_ori)
        token_end = index + len(token)
        token_end_ori = index_ori + len(token)
        if index in entity_start_hash:
            entity_start_hash[index].start = index_ori
        if token_end in entity_end_hash:
            entity = entity_end_hash[token_end]
            entity.end = token_end_ori
            entity.text = original_text[entity.start:entity.end]

        index = token_end
        index_ori = token_end_ori

    annotation.text = original_text
    return annotation


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

    # get BIO labels for one sentence at a time
    predictions = [tagger.tag(xseq) for xseq in featuers]

    # get bio results lines
    bio_lines = make_lines(tokens, predictions)

    # get annotation from bio results
    annotation = bio_to_ann(bio_lines, 'Disease')

    # map back to original text
    annotation = map_to_original(annotation, text)

    return annotation


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('specify text file/folder, model name and result file/folder')
        sys.exit(0)

    text_path = sys.argv[1]
    model_file = sys.argv[2]
    result_file = sys.argv[3]

    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)

    # sentence splitter
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # use WordPunctTokenizer to split disease-suppressor to disease, -, suppressor
    word_punct_tokenizer = WordPunctTokenizer()

    # ann writer
    writer = AnnWriter()

    if os.path.isdir(text_path):
        for root, _, files in os.walk(text_path):
            for f in files:
                if not f.endswith('.txt'):
                    continue

                # filepath ends with .txt
                base_file = f[:-4]
                path = os.path.join(root, f)
                annotation = recognize(path)
                writer.write(os.path.join(result_file, base_file + '.ann'), annotation)

    elif os.path.isfile(text_path):
        if text_path.endswith('.txt'):
            # filepath ends with .txt
            base_file = os.path.basename(text_path)[:-4]
            annotation = recognize(text_path)
            writer.write(result_file, annotation)
        else:
            print('filename must ends with .txt')




