# 682project

## Requirements
Python 3.6
PyTorch 0.4
Scikit-learn 0.20

## Datasets
1. Download [SimpleQuestions dataset](https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz) 
from [Facebook Research](https://research.fb.com/downloads/babi/)


## Run
python src/main.py --model HR-BiLSTM -train -test --hidden_size 512 --optimizer RMSprop --learning_rate 0.001 --dropout 0.3 --margin 0.1 --batch_type batch_question --batch_question_size 1 --data_type WQ

