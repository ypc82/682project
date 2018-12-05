#!/usr/bin/env bash

srun -p 1080ti-short --mem=10GB ${KBQA_ROOT}/bin/run.sh
