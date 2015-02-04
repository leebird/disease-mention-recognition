# disease-mention-recognition
Corpus: http://annotation.dbi.udel.edu/text_mining/corpus/#/NCBI_disease/

Train a model:
```
python -m src.train corpus/BIO/train.bio model/train.model
```

Test with a model: 
```
python -m src.test corpus/BIO/development.bio model/train.model
```

Tag a file or all .txt files in a folder:
```
python -m src.tag corpus/ann/train/2161209.txt model/train.model 2161209.ann
python -m src.tag corpus/ann/train/ model/train.model result/
```

BIO labeling evaluation on dev and test set
```
DEV set      precision    recall  f1-score   support

          B       0.84      0.83      0.84       791
          I       0.91      0.82      0.86      1097

avg / total       0.88      0.83      0.85      1888


TEST set     precision    recall  f1-score   support

          B       0.87      0.81      0.84       961
          I       0.82      0.85      0.84      1087

avg / total       0.85      0.83      0.84      2048
```

Mention level evaluation on dev and test set
```
DEV set:
all entities: 781
level      precision  recall       f1-score
mention    0.82       0.81             0.82
ending     0.91       0.91             0.91

TEST set:
all entities: 955
level      precision  recall       f1-score
mention    0.83       0.78             0.80
ending     0.91       0.85             0.88

```
