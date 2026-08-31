#!/usr/bin/env bash
cd /usr/src

# STT_DEVICE is set to "cuda" by Dockerfile.gpu and unset in the CPU image.
# "$@" comes last so an explicit --device from the command line overrides it.
.venv/bin/python3 -m wyoming_faster_whisper \
    --uri 'tcp://0.0.0.0:10300' --data-dir '/data' \
    --device "${STT_DEVICE:-cpu}" "$@"
