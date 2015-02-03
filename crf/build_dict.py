import nltk
import nltk.data
import os
import sys
from nltk.tokenize import WordPunctTokenizer
import leveldb
import re

if __name__ == '__main__':
    # build bi-gram for a dictionary

    if len(sys.argv) < 4:
        print('specify dictionary file, one name per line, and databases for atom names and bigram')
        sys.exit(0)

    nltk.data.path = ['data']
    dict_file = sys.argv[1]
    db_atom_file = sys.argv[2]
    db_bigram_file = sys.argv[3]

    pattern = re.compile('[a-zA-Z]')

    with open(dict_file, 'r') as handler:

        atom_db = leveldb.LevelDB(db_atom_file)
        bigram_db = leveldb.LevelDB(db_bigram_file)

        # sentence splitter
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        # use WordPunctTokenizer to split disease-suppressor to disease, -, suppressor
        word_punct_tokenizer = WordPunctTokenizer()

        for line in handler:
            line = line.strip().lower()
            # tokenization
            tokens = word_punct_tokenizer.tokenize(line)
            if len(tokens) > 1:
                bigrams = [tokens[i] + '|' + tokens[i + 1] for i in range(len(tokens) - 1)]
                for bigram in bigrams:
                    bigram_db.Put(bigram.encode('utf-8'), ''.encode('utf-8'))
            elif pattern.search(line) is not None:
                atom_db.Put(line.encode('utf-8'), ''.encode('utf-8'))
                

