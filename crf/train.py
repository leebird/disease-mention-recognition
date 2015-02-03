import pycrfsuite
from .utils import sent2features, sent2labels, get_sentences
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('specify BIO file and model name')
        sys.exit(0)

    bio_file = sys.argv[1]
    model_file = sys.argv[2]

    train_sents = list(get_sentences(bio_file))
    
    # print(sent2features(train_sents[0])[0])
    
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    
    trainer = pycrfsuite.Trainer(verbose=False)
    
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    
    trainer.train(model_file)

