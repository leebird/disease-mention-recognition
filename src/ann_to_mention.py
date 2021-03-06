from annotation.readers import AnnReader
from annotation.utils import FileProcessor

import nltk
import nltk.data
import os
import sys
from nltk.tokenize import WordPunctTokenizer
'''ann file to segment file
bcl-2 gene  I-gene
is a gene   O-gene
'''

def ann_to_bio(corpus, bio_file):

    def get_bio_tag(index, entities):
        # get bio tag based on the start index
        for entity in entities:
            if entity.start == index:
                return 2
            elif entity.start < index < entity.end:
                return 2
        return 1

    # sequence output template
    template = '{0} |{1}\n'

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
                if len(sentence.strip()) < 2:
                    continue
                # tokenization
                tokens = word_punct_tokenizer.tokenize(sentence)
                # get pos tag
                pos_tags = nltk.pos_tag(tokens)

                prev_tag = None
                token_list = []

                for pos_tag in pos_tags:
                    token, pos = pos_tag
                    index = text.find(token, index)

                    if index == -1:
                        raise Exception
                    # get bio tag
                    bio_tag = get_bio_tag(index, entities)
                    
                    if prev_tag is not None and prev_tag == bio_tag:
                        token_list.append(token)
                    elif prev_tag is not None and prev_tag != bio_tag:
                        bio_file_handler.write(template.format(' '.join(token_list), prev_tag))
                        prev_tag = bio_tag
                        token_list = [token]
                    elif prev_tag is None:
                        token_list.append(token)
                        prev_tag = bio_tag

                    index += len(token)
                bio_file_handler.write(template.format(' '.join(token_list), prev_tag))
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