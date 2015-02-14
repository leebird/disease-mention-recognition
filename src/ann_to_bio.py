from annotation.readers import AnnReader
from annotation.utils import FileProcessor

import nltk
import nltk.data
import os
import sys
from nltk.tokenize import WordPunctTokenizer
import leveldb
import re

def ann_to_bio(corpus, bio_file):
    def get_bio_tag(index, entities):
        # get bio tag based on the start index
        for entity in entities:
            if entity.start == index:
                return 'B'
            elif entity.start < index < entity.end:
                return 'I'
        return 'O'

    # db and regex
    atom_db = leveldb.LevelDB('data/atom_db')
    upper_number = re.compile(r'^[A-Z]+[0-9]+[A-Z]*$')
    punctuation = re.compile(r'^[^A-Za-z0-9]+$')

    # sequence output template
    template = '{token}\t{pos}\t{pos_2}\t{lower}\t{is_upper}\t' \
               '{is_title}\t{is_first}\t{is_digit}\t{is_punkt}\t' \
               '{is_upper_number}\t{suffix_3}\t{suffix_2}\t{prefix_3}\t{prefix_2}\t' \
               '{in_db}\t{label}\n'

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

                for i, pos_tag in enumerate(pos_tags):
                    token, pos = pos_tag
                    index = text.find(token, index)

                    if index == -1:
                        raise Exception
                    # get bio tag
                    bio_tag = get_bio_tag(index, entities)

                    try:
                        atom_db.Get(token.lower().encode('utf-8'))
                        in_db = True
                    except KeyError:
                        in_db = False
                    
                    is_upper_number = False if upper_number.match(token) is None else True
                    is_punctuation = False if punctuation.match(token) is None else True

                    bio_file_handler.write(template.format(token=token, pos=pos,
                                                           pos_2=pos[:2], lower=token.lower(),
                                                           is_upper=token.isupper(), is_title=token.istitle(),
                                                           is_first=(i == 0),
                                                           is_digit=token.isdigit(), is_punkt=is_punctuation,
                                                           is_upper_number=is_upper_number,
                                                           suffix_3=token[-3:], suffix_2=token[-2:],
                                                           prefix_3=token[:3], prefix_2=token[:2],
                                                           in_db=in_db,
                                                           label=bio_tag))
                    index += len(token)

                # add a newline to separate sentence
                bio_file_handler.write('\n')

    bio_file_handler.close()


if __name__ == '__main__':

    nltk.data.path = ['data']
    

    if len(sys.argv) < 3:
        print('specify ann folder and BIO file')
        sys.exit(0)

    ann_folder = sys.argv[1]
    bio_file = sys.argv[2]

    ann_to_bio(ann_folder, bio_file)

    # corpus_train = 'corpus/ann/train'
    # corpus_test = 'corpus/ann/test'
    # corpus_dev = 'corpus/ann/development'
    #
    # bio_train = 'corpus/BIO/train.bio'
    # bio_test = 'corpus/BIO/test.bio'
    # bio_dev = 'corpus/BIO/development.bio'
    #
    # ann_to_bio(corpus_train, bio_train)
    # ann_to_bio(corpus_dev, bio_dev)
    # ann_to_bio(corpus_test, bio_test)