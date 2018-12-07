#!/usr/bin/env bash

srun -p 1080ti-short --mem=10GB --gres=gpu:1 ${KBQA_ROOT}/bin/run.sh
