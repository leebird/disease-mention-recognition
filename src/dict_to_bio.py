import nltk
import nltk.data
import os
import sys
from nltk.tokenize import WordPunctTokenizer

nltk.data.path = ['data']

def ann_to_bio(text_file, bio_file):

    # sequence output template
    template = '{0}\t{1}\t{2}\n'

    # sentence splitter
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # use WordPunctTokenizer to split disease-suppressor to disease, -, suppressor
    word_punct_tokenizer = WordPunctTokenizer()

    # open the bio file
    bio_file_handler = open(bio_file, 'a')
    
    # open the text file
    text_file_handler = open(text_file, 'r')
    
    for line in text_file_handler:
        line = line.strip()

        # tokenization
        tokens = word_punct_tokenizer.tokenize(line)
        
        if len(tokens) < 2:
            continue
        
        # get pos tag
        pos_tags = nltk.pos_tag(tokens)

        for i, pos_tag in enumerate(pos_tags):
            token, pos = pos_tag

            # get bio tag
            if i == 0:
                bio_tag = 'B'
            else:
                bio_tag = 'I'
            bio_file_handler.write(template.format(token, pos, bio_tag))

        # add a newline to separate sentence
        bio_file_handler.write('\n')

    bio_file_handler.close()
    text_file_handler.close()

dict_medic = 'old/MEDIC/disease_names.txt'
dict_umls = 'old/UMLS/api/disease_names.txt'
dict_umls_atom = 'old/UMLS/api/disease_names_atom.txt'

bio_medic = 'corpus/BIO/medic.bio'
bio_umls = 'corpus/BIO/umls.bio'
bio_umls_atom = 'corpus/BIO/umls_atom.bio'

ann_to_bio(dict_medic, bio_medic)
ann_to_bio(dict_umls, bio_umls)
ann_to_bio(dict_umls_atom, bio_umls_atom)