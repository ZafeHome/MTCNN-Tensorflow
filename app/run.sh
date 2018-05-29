#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
mkdir -p ./logs

if [ -z ${GUNICORN_WORKERS+x} ]; then GUNICORN_WORKERS=1; fi
if [ -z ${GUNICORN_TIMEOUT+x} ]; then GUNICORN_TIMEOUT=180; fi

gunicorn app:app \
    --workers ${GUNICORN_WORKERS} \
    --bind 0.0.0.0:5000 \
    --log-file=./logs/gunicorn.log \
    --log-level=DEBUG \
    --timeout=${GUNICORN_TIMEOUT}
