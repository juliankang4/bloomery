// The set writer of the mainline oracle dumpers (tools/ref/clefvis/dump_mtmd.cpp, tools/ref/emb2/dump_emb2.cpp): sha256,
// the reader of the id files the dumpers take, the raw f32 and i32 rows of dump_ref's set format, and the manifest with its
// `# complete` trailer, renamed into place last. The includer defines DUMP_TOOL, the name its messages open with, before
// including this file; DIE prints one and exits 1.
#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifndef DUMP_TOOL
#error "define DUMP_TOOL, the name every message opens with, before including set_writer.h"
#endif

#define DIE(...) do { fprintf(stderr, DUMP_TOOL ": " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

// ---- sha256 -----------------------------------------------------------------------------------------------------

struct sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t  buf[64];
    size_t   fill = 0;
    uint64_t total = 0;
    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const uint8_t * p, size_t n) {
        total += n;
        while (n > 0) {
            const size_t take = std::min(n, 64 - fill);
            memcpy(buf + fill, p, take);
            fill += take; p += take; n -= take;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

static std::string sha256_of(const uint8_t * p, size_t n) {
    sha256 s;
    s.update(p, n);
    return s.hex();
}

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

// The ids of an id file (clefvis.sh --ids, the prose ids, crates/tokenizer/tools/oracle.sh): one base-10 id from 0 to
// INT32_MAX a line, digits only. Any other line, an empty one or one with a NUL byte included, is a DIE naming the file and the line.
static std::vector<int32_t> parse_ids(const std::string & path, const std::vector<uint8_t> & raw) {
    std::vector<int32_t> ids;
    std::istringstream in(std::string(raw.begin(), raw.end()));
    std::string line;
    long n = 0;
    while (std::getline(in, line)) {
        ++n;
        char * end = nullptr;
        errno = 0;
        const long v = line.empty() || line[0] < '0' || line[0] > '9' ? -1 : strtol(line.c_str(), &end, 10);
        if (v < 0 || end != line.c_str() + line.size() || errno != 0 || v > INT32_MAX) {
            DIE("%s:%ld is not a token id: '%s'", path.c_str(), n, line.c_str());
        }
        ids.push_back((int32_t) v);
    }
    return ids;
}

// ---- the set writer (dump_ref's format) ---------------------------------------------------------------------------

static std::string safe_name(const std::string & s) {
    std::string r = s;
    for (char & c : r) if (c == '/' || c == '\\' || c == ' ') c = '_';
    return r;
}

static void write_raw(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) DIE("cannot write %s", path.c_str());
    if (fwrite(data, 1, bytes, f) != bytes) DIE("short write on %s", path.c_str());
    if (fclose(f) != 0) DIE("cannot flush %s", path.c_str());
}

struct set_writer {
    std::string dir;
    std::string head;     // header lines
    std::string rows;     // tensor and int rows
    int         written = 0;
    std::map<std::string, int> occ;

    explicit set_writer(const std::string & d) : dir(d) { mkdir(dir.c_str(), 0755); }

    void line(const std::string & l) { head += l + "\n"; }

    // One f32 tensor row and its file.
    void tensor_f32(const std::string & name, const char * op, const int64_t ne[4], const float * data) {
        const int o = occ[name]++;
        const size_t n = (size_t) ne[0] * ne[1] * ne[2] * ne[3];
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(data[i])) DIE("%s: value %zu is not finite", name.c_str(), i);
            sum += data[i];
        }
        write_raw(dir + "/" + safe_name(name) + "." + std::to_string(o) + ".f32", data, n * sizeof(float));
        char buf[512];
        snprintf(buf, sizeof(buf), "tensor\t%s\t%d\tf32\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t1\t0\t-\t-\n", name.c_str(), o,
                 (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], n * sizeof(float), sum, op);
        rows += buf;
        written++;
    }

    // An i32 tensor: the f32 file (exact below 2^24, which the writer checks), the lossless i32 twin and the `int` row.
    void tensor_i32(const std::string & name, const char * op, const int64_t ne[4], const std::vector<int32_t> & v) {
        const int o = occ[name]++;
        const size_t n = (size_t) ne[0] * ne[1] * ne[2] * ne[3];
        if (v.size() != n) DIE("%s: %zu ints for shape %lld x %lld", name.c_str(), v.size(), (long long) ne[0], (long long) ne[1]);
        std::vector<float> f(n);
        int64_t sum = 0;
        uint64_t absmax = 0;
        for (size_t i = 0; i < n; ++i) {
            if (v[i] < -(1 << 24) || v[i] > (1 << 24)) DIE("%s: %d does not widen to f32 exactly", name.c_str(), v[i]);
            f[i] = (float) v[i];
            sum += v[i];
            absmax = std::max<uint64_t>(absmax, (uint64_t) (v[i] < 0 ? -(int64_t) v[i] : v[i]));
        }
        const std::string stem = safe_name(name) + "." + std::to_string(o);
        write_raw(dir + "/" + stem + ".f32", f.data(), n * sizeof(float));
        write_raw(dir + "/" + stem + ".i32", v.data(), n * sizeof(int32_t));
        char buf[512];
        snprintf(buf, sizeof(buf), "tensor\t%s\t%d\ti32\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t1\t0\t-\t-\n", name.c_str(), o,
                 (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], n * sizeof(float), (double) sum, op);
        rows += buf;
        snprintf(buf, sizeof(buf), "int\t%s\t%d\ttensor\ti32\ti32\tflat\t%zu\t%zu\t%lld\t%llu\t%s.i32\n", name.c_str(), o, n,
                 n * sizeof(int32_t), (long long) sum, (unsigned long long) absmax, stem.c_str());
        rows += buf;
        written++;
    }

    // The manifest, the trailer, and the rename that publishes the set.
    void finish() {
        const std::string final_path = dir + "/MANIFEST.tsv", partial = final_path + ".partial";
        FILE * m = fopen(partial.c_str(), "w");
        if (!m) DIE("cannot write %s", partial.c_str());
        fputs(head.c_str(), m);
        fputs("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n", m);
        fputs("# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\n", m);
        fputs(rows.c_str(), m);
        fprintf(m, "# complete\t%d\t0\n", written);
        if (fclose(m) != 0 || rename(partial.c_str(), final_path.c_str()) != 0) DIE("cannot install %s", final_path.c_str());
        fprintf(stderr, DUMP_TOOL ": %d rows into %s\n", written, dir.c_str());
    }
};
