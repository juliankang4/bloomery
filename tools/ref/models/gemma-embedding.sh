#!/usr/bin/env bash
# shellcheck shell=bash
# EmbeddingGemma v1's text-only oracle, using the same pinned mainline as v2.
# shellcheck disable=SC2034
MODEL_NAME=gemma-embedding
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/embeddinggemma-300M/embeddinggemma-300M-Q8_0.gguf}
MODEL_SHA256=b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63
MMPROJ='' MMPROJ_SHA256=''
: "${LCPP:=/home/user/llama.cpp-6868a52c}"
LCPP_PIN=6868a52cb337450658006ad2f9281c2eb63f1666
IMAGES=()
EMBED_VOCAB=llama
EMBED_CASES=cases-emb1.txt
EMBED_CONTEXT=2048
EMBED_STEPS=(text server text.cpu)
