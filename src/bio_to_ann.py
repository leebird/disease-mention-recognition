from annotation.annotation import Annotation
from annotation.writers import AnnWriter
from annotation.utils import FileProcessor

import os
import sys

def bio_to_ann(bio_lines, tag_type):
    # annotation
    annotation = Annotation()

    index = 0

    if tag_type == 'BIO':
        # keep BIO label for debugging
        for line in bio_lines:
            line = line.strip()

            if len(line) == 0:
                annotation.text += '\n'
                index += 1
                continue

            fields = line.split('\t')
            token, label = fields[0], fields[-1]
            annotation.text += token + ' '
            if label != 'O':
                annotation.add_entity(label, index, index + len(token), token)
            index += 1 + len(token)

    elif tag_type == 'Disease':

        # transfer BIO to disease mention
        start = -1
        end = -1

        for line in bio_lines:
            line = line.strip()

            if len(line) == 0:
                annotation.text += '\n'
                index += 1
                continue

            fields = line.split('\t')
            token, label = fields[0], fields[-1]
            
            # so that we can use index + len(token) for the token end
            annotation.text += token + ' '

            # index is the current position in text
            if label == 'B':
                start = index
                end = index + len(token)
            elif label == 'I':
                end = index + len(token)
            elif label == 'O' and start > -1:
                annotation.add_entity('Disease', start, end, annotation.text[start:end])
                start = -1
                end = -1

            index += 1 + len(token)

    annotation.text = annotation.text.strip()
    return annotation


def bio_to_ann_file(bio_file, text_file, ann_file, tag_type):
    # ann writer
    writer = AnnWriter()

    # open the bio file
    bio_file_handler = open(bio_file, 'r')

    annotation = bio_to_ann(bio_file_handler, tag_type)

    bio_file_handler.close()
    writer.write(ann_file, annotation)
    FileProcessor.write_file(text_file, annotation.text)


if __name__ == '__main__':
    tag_type = 'Disease'

    if len(sys.argv) < 4:
        print('specify BIO, txt and ann files')
        sys.exit(0)

    bio_file = sys.argv[1]
    text_file = sys.argv[2]
    ann_file = sys.argv[3]

    bio_to_ann_file(bio_file, text_file, ann_file, tag_type)

    # bio_file = 'data/result/result.bio'
    # text_file = 'data/result/result.txt'
    # ann_file = 'data/result/result.ann'
    # 
    # bio_to_ann(bio_file, text_file, ann_file, tag_type)
    #
    # bio_file = 'corpus/BIO/development.bio'
    # text_file = 'data/result/gold.txt'
    # ann_file = 'data/result/gold.ann'
    #
    # bio_to_ann(bio_file, text_file, ann_file, tag_type)