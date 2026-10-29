#!/bin/bash
# Probe an already-running, unauthenticated server's framing with raw socket requests.
# Usage: bash tools/ref/serve-framing.sh HOST PORT
# The same client accepts the mock seat or a model-backed bloomery-serve seat.
set -eu
python3 - "$@" <<'PY'
import http.client
import socket
import sys

if len(sys.argv) != 3:
    sys.exit("usage: serve-framing.sh HOST PORT")
host, port = sys.argv[1], int(sys.argv[2])
body = b'{"content":"x"}'
n = len(body)
next_request = b'GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n'


class WireSocket(socket.socket):
    def makefile(self, mode="r", buffering=None, **kwargs):
        # A one-byte buffer fills short body reads and never reads past the current response.
        return super().makefile(mode, buffering=1, **kwargs)


def probe(name, headers, wire, status=400, reusable=False):
    with WireSocket() as sock:
        sock.settimeout(10)
        sock.connect((host, port))
        sock.sendall(b'POST /tokenize HTTP/1.1\r\nHost: localhost\r\n' + headers + b'\r\n' + wire + next_request)
        sock.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(sock)
        response.begin()
        response.read()
        connection = 'keep-alive' if reusable else 'close'
        assert response.status == status, (name, response.status)
        assert response.getheader('Connection') == connection, (name, response.getheaders())
        if reusable:
            response = http.client.HTTPResponse(sock)
            response.begin()
            response.read()
            assert response.status == 200, (name, 'next request', response.status)
        try:
            assert sock.recv(1) == b'', (name, 'unexpected pipelined response')
        except ConnectionResetError:
            pass
    print(f'PASS {name} status={status} connection={connection} next={200 if reusable else "closed"}', flush=True)


for name, headers, wire in [
    ('conflicting-lengths', f'Content-Length: {n}\r\ncontent-length: {n+1}\r\n'.encode(), body),
    ('conflicting-comma-lengths', f'Content-Length: {n}, {n+1}\r\n'.encode(), body),
    ('scalar-vtab', f'Content-Length: \x0b{n}\r\n'.encode(), body),
    ('comma-vtab', f'Content-Length: {n},\x0b{n}\r\n'.encode(), body),
    ('chunk-vtab', b'Transfer-Encoding: chunked\r\n', f'{n:x}\x0b\r\n'.encode() + body + b'\r\n0\r\n\r\n'),
    ('chunk-data-junk', b'Transfer-Encoding: chunked\r\n', f'{n:x}\r\n'.encode() + body + b'XX0\r\n\r\n'),
]:
    probe(name, headers, wire)
probe('equal-lengths', f'Content-Length: {n}\r\ncontent-length: {n}\r\n'.encode(), body, 200, True)
probe('http-whitespace', f'Content-Length: \t{n} ,\t{n}\t\r\n'.encode(), body, 200, True)
probe('chunk-extension', b'Transfer-Encoding: chunked\r\n', f'{n:x} \t;x=y\r\n'.encode() + body + b'\r\n0\r\nTrailer: value\r\n\r\n', 200, True)
probe('chunked-plus-length', b'Transfer-Encoding: chunked\r\nContent-Length: 1\r\n', f'{n:x}\r\n'.encode() + body + b'\r\n0\r\n\r\n', 200)
PY
