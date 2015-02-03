from annotation.readers import AnnReader
from annotation.utils import FileProcessor

import nltk
import nltk.data
import os
import sys
from nltk.tokenize import WordPunctTokenizer

nltk.data.path = ['data']


def ann_to_bio(corpus, bio_file):

    def get_bio_tag(index, entities):
        # get bio tag based on the start index
        for entity in entities:
            if entity.start == index:
                return 'B'
            elif entity.start < index < entity.end:
                return 'I'
        return 'O'

    # sequence output template
    template = '{0}\t{1}\t{2}\n'

    # sentence splitter
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # use WordPunctTokenizer to split disease-suppressor to disease, -, suppressor
    word_punct_tokenizer = WordPunctTokenizer()

    # ann reader
    reader = AnnReader()

    # open the bio file
    bio_file_handler = open(bio_file, 'a')

    for root, _, files in os.walk(corpus):
        for file in files:
            if not file.endswith('.ann'):
                continue

            pmid = file[:-4]
            annotation = reader.parse_file(os.path.join(root, file))
            entities = annotation.entities
            entities = sorted(entities, key=lambda a: a.start)
            text = FileProcessor.read_file(os.path.join(root, pmid + '.txt'))
            sentences = sent_detector.tokenize(text.strip())
            index = 0

            for sentence in sentences:
                # tokenization
                tokens = word_punct_tokenizer.tokenize(sentence)
                # get pos tag
                pos_tags = nltk.pos_tag(tokens)

                for pos_tag in pos_tags:
                    token, pos = pos_tag
                    index = text.find(token, index)

                    if index == -1:
                        raise Exception
                    # get bio tag
                    bio_tag = get_bio_tag(index, entities)
                    bio_file_handler.write(template.format(token, pos, bio_tag))
                    index += len(token)

                # add a newline to separate sentence
                bio_file_handler.write('\n')

    bio_file_handler.close()

corpus_train = 'corpus/ann/train'
corpus_test = 'corpus/ann/test'
corpus_dev = 'corpus/ann/development'

bio_train = 'corpus/BIO/train.bio'
bio_test = 'corpus/BIO/test.bio'
bio_dev = 'corpus/BIO/development.bio'

ann_to_bio(corpus_train, bio_train)
ann_to_bio(corpus_dev, bio_dev)
ann_to_bio(corpus_test, bio_test)