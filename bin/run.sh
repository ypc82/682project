#!/usr/bin/env bash
set -x

python3 ${KBQA_ROOT}/src/main.py --model HR-BiLSTM -train -test --hidden_size 50 --optimizer Adam --learning_rate 0.1 --dropout 0.3 --margin 0.1 --batch_type batch_question --batch_question_size 1 --data_type WQ

