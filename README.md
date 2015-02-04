# disease-mention-recognition
Corpus: http://annotation.dbi.udel.edu/text_mining/corpus/#/NCBI_disease/

BIO labeling evaluation on dev and test set
```
DEV set         precision    recall  f1-score   support

          B       0.84      0.82      0.83       791
          I       0.92      0.81      0.86      1097

avg / total       0.89      0.81      0.85      1888

TEST set        precision    recall  f1-score   support

          B       0.87      0.79      0.82       961
          I       0.87      0.80      0.83      1087

avg / total       0.87      0.79      0.83      2048
```

Mention level evaluation on dev and test set
```
DEV set:
all entities: 781

Exact match:
precision: 0.82
recall: 0.80
f-score: 0.81

Same-ending match:
precision: 0.91
recall: 0.90
f-score: 0.91

------------------

TEST set:
all entities: 955

Exact match:
precision: 0.84
recall: 0.76
f-score: 0.80

Same-ending match:
precision: 0.92
recall: 0.83
f-score: 0.87

```
