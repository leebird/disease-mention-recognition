import pycrfsuite
from .utils import sent2features, sent2labels, bio_classification_report, get_sentences, sent2tokens
import sys
import os
from annotation.utils import FileProcessor

root = os.path.dirname(os.path.dirname(__file__))
legonlp = os.path.join(root, 'data/legonlp-master')
sys.path.append(legonlp)

from .bio_to_ann import bio_to_ann, bio_to_ann_file
from .evaluate import Evaluation


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('specify BIO file, model name and output file')
        sys.exit(0)

    bio_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)

    test_sents = list(get_sentences(bio_file))
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    token_test = [sent2tokens(s) for s in test_sents]

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    # evaluate BIO lablelling
    print(bio_classification_report(y_test, y_pred))

    # evaluate mention recognition
    bio_lines = []
    for tokens, labels in zip(token_test, y_pred):
        for token, label in zip(tokens, labels):
            bio_lines.append(token + '\t' + label + '\n')
        bio_lines.append('\n')
    
    FileProcessor.write_file(output_file, ''.join(bio_lines))
    
    user_annotation = bio_to_ann(bio_lines, 'Disease')
    with open(bio_file, 'r') as bio_file_handler:
        golden_annotation = bio_to_ann(bio_file_handler, 'Disease')
        Evaluation.evaluate({'test': user_annotation}, {'test': golden_annotation}, 'mention')
        Evaluation.evaluate({'test': user_annotation}, {'test': golden_annotation}, 'ending')

        # from collections import Counter
        # info = tagger.info()
        #
        # print("Top likely transitions:")
        # print_transitions(Counter(info.transitions).most_common(15))
        #
        # print("\nTop unlikely transitions:")
        # print_transitions(Counter(info.transitions).most_common()[-15:])
        #
        # print("Top positive:")
        # print_state_features(Counter(info.state_features).most_common(20))
        #
        # print("\nTop negative:")
        # print_state_features(Counter(info.state_features).most_common()[-20:])