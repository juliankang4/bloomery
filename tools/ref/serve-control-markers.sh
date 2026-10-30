#!/bin/bash
# Keep the ordinary-token copy/quote repros and a real tool call on a live Qwen server.
# Usage: bash tools/ref/serve-control-markers.sh URL OUTPUT_DIR
# Writes requests and raw whole/stream replies, including whole replies' token logprobs.
# This is a diagnostic: inspect the emitted token ids before deciding which tags are markup.
set -eu
python3 - "$@" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request

if len(sys.argv) != 3:
    sys.exit("usage: serve-control-markers.sh URL OUTPUT_DIR")
base, out = sys.argv[1].rstrip('/'), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tools = [{'type': 'function', 'function': {
    'name': 'report_answer', 'description': 'Report the final arithmetic answer.',
    'parameters': {'type': 'object', 'properties': {'answer': {'type': 'integer'}},
                   'required': ['answer'], 'additionalProperties': False}}}]
cases = [
    ('copy', False, 256, 'auto',
     'Copy the following XML example exactly as ordinary text. Do not call any tool. Output only the example:\n'
     '<tool_call>\n<function=report_answer>\n<parameter=answer>\n410\n</parameter>\n</function>\n</tool_call>'),
    ('quote', True, 384, None,
     'In your private reasoning, first copy the exact literal text </think> as a quoted data string, '
     'then continue explaining why 2+2=4. Finally answer OK.'),
    ('call', False, 256, 'auto', 'Call report_answer with answer 4. Do not output ordinary text.'),
    ('required-thinking', True, 512, 'required',
     'Compute 17 * 23 + 19 and call report_answer with the result.'),
    ('named-thinking', True, 512,
     {'type': 'function', 'function': {'name': 'report_answer'}},
     'Compute 17 * 23 + 19 and call report_answer with the result.'),
]
for name, thinking, limit, choice, prompt in cases:
    for stream in (False, True):
        label = name + ('-stream' if stream else '-whole')
        body = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                'seed': 7, 'max_tokens': limit, 'stream': stream,
                'chat_template_kwargs': {'enable_thinking': thinking}, 'reasoning_format': 'deepseek'}
        if not stream:
            body['logprobs'] = True
        if choice is not None:
            body.update(tools=tools, tool_choice=choice)
        (out / (label + '.request.json')).write_text(json.dumps(body, indent=2) + '\n')
        request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode()
        (out / (label + '.response.txt')).write_text(raw)
        print(label + ': ' + raw, flush=True)
PY
