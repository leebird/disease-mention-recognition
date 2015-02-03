from annotation.annotation import Annotation
from annotation.writers import AnnWriter
from annotation.utils import FileProcessor

import os
import sys


def bio_to_ann(bio_file, text_file, ann_file, tag_type):
    # ann writer
    writer = AnnWriter()

    # annotation
    annotation = Annotation()

    # open the bio file
    bio_file_handler = open(bio_file, 'r')
    index = 0

    if tag_type == 'BIO':
        # keep BIO label for debugging
        for line in bio_file_handler:
            line = line.strip()
    
            if len(line) == 0:
                annotation.text += '\n'
                index += 1
                continue
    
            fields = line.split('\t')
            token, label = fields[0], fields[-1]
            annotation.text += ' ' + token
            if label != 'O':
                annotation.add_entity(label, index + 1, index + 1 + len(token), token)
            index += 1 + len(token)
            
    elif tag_type == 'Disease':
        
        # transfer BIO to disease mention
        start = -1
        end = -1
        
        for line in bio_file_handler:
            line = line.strip()

            if len(line) == 0:
                annotation.text += '\n'
                index += 1
                continue
            
            fields = line.split('\t')
            token, label = fields[0], fields[-1]
            annotation.text += ' ' + token
            
            if label == 'B':
                start = index + 1
                end = index + 1 + len(token)
            elif label == 'I':
                end += 1 + len(token)
            elif label == 'O' and start > -1:
                annotation.add_entity('Disease', start, end, annotation.text[start:end])
                start = -1
                end = -1
                
            index += 1 + len(token)

    bio_file_handler.close()
    writer.write(ann_file, annotation)
    FileProcessor.write_file(text_file, annotation.text)

if __name__ == '__main__':
    tag_type = 'Disease'
    
    bio_file = 'data/result/result.bio'
    text_file = 'data/result/result.txt'
    ann_file = 'data/result/result.ann'

    bio_to_ann(bio_file, text_file, ann_file, tag_type)

    bio_file = 'corpus/BIO/test.bio'
    text_file = 'data/result/gold.txt'
    ann_file = 'data/result/gold.ann'

    bio_to_ann(bio_file, text_file, ann_file, tag_type)