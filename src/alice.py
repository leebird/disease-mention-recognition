#!/usr/bin/env python
# -*- coding: utf-8 -*-

from suds.xsd.doctor import Import
from suds.xsd.doctor import ImportDoctor
from suds.client import Client
import nltk
import nltk.data
import re
import os
from annotation.readers import AnnReader
from annotation.utils import FileProcessor
import time
import leveldb

class Allie(object):
    def __init__(self):
        # https://fedorahosted.org/suds/wiki/Documentation#FIXINGBROKENSCHEMAs
        imp = Import('http://schemas.xmlsoap.org/soap/encoding/', 'http://schemas.xmlsoap.org/soap/encoding/')
        wsdl = 'file:///home/leebird/Downloads/allie.wsdl'
        self.client = Client(wsdl, plugins=[ImportDoctor(imp)])


    def match_by_longform(self, longform):
        pairInfo = self.client.service.GetPairsByLongform(param0=longform)
        shortforms = []
        for pair in pairInfo:
            shortforms.append(str(pair.abbreviation))
        shortforms = [sf.lower() for sf in shortforms]
        print(shortforms)
        time.sleep(2)
        return shortforms
        # print("PairInfo["+str(pair.pair_id)+"|"+str(pair.abbreviation)+"|"+str(pair.long_form)+"]")

class AllieLocal(object):
    def __init__(self):
        self.db = leveldb.LevelDB('/home/leebird/Projects/disease/data/allie_db')

    def match_by_longform(self, longform):
        try:
            shortforms = self.db.Get(longform.encode('utf-8')).decode('utf-8').split('|')
            print(shortforms)
            return shortforms
        except KeyError:
            return []

class PostProcessor(object):
    def __init__(self):
        # self.allie = Allie()
        self.allie = AllieLocal()
        nltk.data.path = ['data']
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.pattern = re.compile(r'\(([^\(\)]+?)\)')

    def get_prev_tokens(self, text, offset):
        token = ''
        tokens = []
        count = 0
        for i in range(offset, -1, -1):
            if count > 9:
                break
            if text[i] == ' ':
                if len(token) == 0:
                    continue
                else:
                    tokens.append(token)
                    token = ''
                    count += 1
            else:
                token += text[i]
        if count < 10 and len(token) > 0:
            tokens.append(token)
        tokens.reverse()
        return tokens

    def get_shortforms(self, tokens):
        for i in range(0, len(tokens)):
            if tokens[i] == 'of':
                break
            longform = ' '.join(tokens[i:])
            longform = longform.replace(' -', '-')
            longform = longform.replace('- ', '-')
            shortforms = self.allie.match_by_longform(longform.lower())
            if len(shortforms) > 0:
                return shortforms
        return []

    def get_longform_end(self, text, offset):
        for i in range(offset, -1, -1):
            if text[i] == ' ':
                continue
            else:
                return i + 1

    def is_longform_disease(self, annotation, end):
        for entity in annotation.entities:
            if entity.end == end:
                return entity.text
        return False

    def is_shortform_disease(self, annotation, start, end):
        for entity in annotation.entities:
            if entity.start >= start and entity.end <= end:
                return True
        return False

    def is_overlap(self, annotation, start, end):
        for entity in annotation.entities:
            if entity.start < end and entity.end > start:
                return True
        return False

    def mark_all_shortforms(self, annotation, acronym):
        ac_pattern = re.compile(re.escape(acronym))
        matches = ac_pattern.finditer(annotation.text)
        for match in matches:
            start = match.start()
            end = match.end()
            if start > 0 and annotation.text[start - 1] != ' ':
                continue
            if end < len(annotation.text) and annotation.text[end] != ' ':
                continue
            if not self.is_overlap(annotation, start, end):
                print('adding new entity')
                annotation.add_entity('Disease', start, end, acronym)

    def remove_all_shortforms(self, annotation, acronym):
        to_remove = []
        for entity in annotation.entities:
            if entity.text.lower == acronym.lower():
                print('remove entity')
                to_remove.append(entity)

        for entity in to_remove:
            annotation.remove(entity)

    def process(self, annotations):
        for pmid, annotation in annotations.items():
            if len(annotation.entities) == 0:
                continue

            # sentences = self.sent_detector.tokenize(annotation.text)
            # sent_offset = 0
            # for sentence in sentences:
            
            text = annotation.text
            in_paren = self.pattern.finditer(text)
            for match in in_paren:
                acronym = match.group(1).strip()
                start = match.start(0)
                end = match.end(0)
                is_sf_disease = self.is_shortform_disease(annotation, start, end)

                longform_end = self.get_longform_end(text, start-1)
                is_lf_disease = self.is_longform_disease(annotation, longform_end)

                if is_lf_disease and is_sf_disease:
                    print(pmid+'\tboth true: ' + is_lf_disease + '\t' + acronym)
                    self.mark_all_shortforms(annotation, acronym)

                if not is_sf_disease and is_lf_disease:
                    print(pmid+'\tlf but not sf: ' + is_lf_disease + '\t' + acronym)
                    lf_tokens = is_lf_disease.lower().split()
                    shortforms = self.get_shortforms(lf_tokens)
                    if acronym.lower() in shortforms:
                        self.mark_all_shortforms(annotation, acronym)

                if is_sf_disease and not is_lf_disease:
                    print(pmid+'\tsf but not lf: ' + str(is_lf_disease) + '\t' + acronym)
                    self.remove_all_shortforms(annotation, acronym)

if __name__ == '__main__':
    # allie = Allie()
    # shorts = allie.match_by_longform('Myotonic dystrophy')
    # print(shorts)

    reader = AnnReader()
    annotations = reader.parse_folder('corpus/ann/development', '.ann')
    for pmid, annotation in annotations.items():
        txt = os.path.join('corpus/ann/development/', pmid + '.txt')
        text = FileProcessor.read_file(txt)
        annotation.text = text

    pp = PostProcessor()
    pp.process(annotations)