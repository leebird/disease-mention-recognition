from annotation.readers import SGMLReader
from annotation.writers import AnnWriter
from annotation.utils import FileProcessor
import re

def change_format(corpus, corpus_to):
    corpus_file = open(corpus, 'r')
    
    def tag_handler(tag):
        pattern = re.compile(r'(.+?)="(.+?)"')
        match = pattern.search(tag)
        if match is not None:
            tag_info = {}
            tag_info['tag'] = match.group(1)
            tag_info['category'] = match.group(2)
            return tag_info
        
    
    reader = SGMLReader(tag_handler=tag_handler)
    writer = AnnWriter()
    
    for line in corpus_file:
        tokens = line.split('\t')
        pmid, abstract = tokens[0], ' '.join(tokens[1:])
        annotation = reader.parse(abstract)
        writer.write(corpus_to+pmid+'.ann',annotation)
        FileProcessor.write_file(corpus_to+pmid+'.txt',annotation.text)


corpus_train = 'corpus/original/NCBI_corpus_training.txt'
corpus_test = 'corpus/original/NCBI_corpus_testing.txt'
corpus_dev = 'corpus/original/NCBI_corpus_development.txt'

corpus_train_to = 'corpus/ann/train/'
corpus_test_to = 'corpus/ann/test/'
corpus_dev_to = 'corpus/ann/development/'

change_format(corpus_train, corpus_train_to)
change_format(corpus_test, corpus_test_to)
change_format(corpus_dev, corpus_dev_to)