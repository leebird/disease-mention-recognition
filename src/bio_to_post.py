import nltk
import nltk.data
import sys

def write_bio(tokens, pos_tags, bio_labels, handler):
    template = '{0}\t{1}\t{2}\n'
    for token, pos, label in zip(tokens, pos_tags, bio_labels):
        line = template.format(token, pos, label)
        handler.write(line)
    handler.write('\n')

def bio_to_post_file(bio_file, bio_post_file):
    # open the bio file
    with open(bio_file, 'r') as bio_file_handler, \
            open(bio_post_file, 'a') as bio_post_file_handler:
        tokens = []
        labels = []
        for line in bio_file_handler:
            if len(line.strip()) == 0:
                # get pos tag
                pos_tags = nltk.pos_tag(tokens)
                pos_tags = [tag[1] for tag in pos_tags]
                write_bio(tokens, pos_tags, labels, bio_post_file_handler)
                tokens, labels = [], []
            elif line.strip().startswith('#'):
                continue
            else:
                token, label = line.strip().split('\t')
                tokens.append(token)
                labels.append(label)
        
if __name__ == '__main__':
    nltk.data.path = ['data']
    
    if len(sys.argv) < 3:
        print('specify BIO and post-BIO files')
        sys.exit(0)

    bio_file = sys.argv[1]
    post_bio_file = sys.argv[2]

    bio_to_post_file(bio_file, post_bio_file)
    
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