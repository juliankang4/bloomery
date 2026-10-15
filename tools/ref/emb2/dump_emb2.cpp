// dump_emb2: llama.cpp mainline's EmbeddingGemma v1 and v2 oracle sets in dump_ref's format
// (tools/ref/clefvis/set_writer.h). The architecture's refset families check them; v2 also has the gemma4v tower.
//
//   dump_emb2 text  -m <gguf> --ids-dir <dir> --case <id> ... --out <dir> --card <name> --build-record <file>
//                   [--cpu] [-t <threads>]
//   dump_emb2 tower -m <gguf> --mmproj <gguf> --image <name>=<png> ... --out <dir> --card <name> --build-record <file>
//                   [--cpu] [-t <threads>]
//
// text. Per case, the ids of <ids-dir>/<id>.ids (one id per line, as the tokenizer oracle writes them,
//   crates/tokenizer/tools/oracle.sh with the profile's vocabulary: llama-tokenize's default, BOS and EOS included)
//   go through one llama_decode, every position in one ubatch (the model is non-causal; llama-server
//   refuses an input longer than its ubatch). Three contexts, each with embeddings on:
//     clean   pooling none, no callback: llama_get_embeddings_ith of every position is `result_embd` (the output
//             projection of `result_norm` in v2, the dense head's per-token output in v1). Rows `<case>/embd` [768, n].
//     pooled  pooling mean (the file's pooling_type and llama-server's): llama_get_embeddings_seq, before any
//             L2 normalization, after v1's learned dense head. Row `<case>/pooled` [768, 1].
//     taps    pooling none, a cb_eval that asks for the nodes TEXT_TAPS names: `inp_scaled` (token rows times
//             sqrt(n_embd)), `inp_per_layer` (the PLE input, [per_layer, layers, n]), `l_out-<il>` of every layer,
//             `result_norm`, and the output projection, written as `result_embd` (v1 has no `inp_per_layer`): with pooling none the graph's pooling
//             step passes t_embd through and renames that node `result_embd_pooled`. Asking for a node splits the
//             scheduler's graph, so `# tap_effect` says whether the taps pass's `result_embd` equals the clean pass's
//             bit for bit; the clean rows are the ones to compare against.
//   `# model` is the text file, `# arch` its architecture.
// tower. Per image, mtmd's encode of the image chunk through a context with no callback (`<image>/embd`
//   [n_embd, tokens], mtmd_get_output_embd: what llama-server hands the text model) and through a second context
//   whose cb_eval asks for the nodes TOWER_TAPS names (gemma4v.cpp and clip.cpp's build_vit): `inp_raw_scaled` (the
//   preprocessed image, 2x - 1, [W, H, 3, 1]), `pos_embd`, `layer_out-<il>` of every block, `pooled` (the 3x3 average
//   times sqrt(n_embd)), `std_scaled` (when the projector has std_bias and std_scale), `projected`. `# image` lines
//   carry the png's sha256, the canvas and the token count; `# tap_effect` compares `projected` with the clean `embd`.
//   As dump_mtmd's tower sets, `# model` is the projector file and `# arch` its architecture (`clip`); the text file is
//   `# text_model`, and `# mmproj` carries the projector's sha256.
//
// Rows are `tensor` rows named `<case or image>/<node>`, occurrence 0, raw little-endian f32 in ggml order (ne0
// contiguous). The header carries what the node-dump reader checks (`# model`, `# arch`, `# build` from
// BLOOMERY_REF_BUILD, the `# complete` trailer, written last) and what the family's own reader checks: `# emb2` (text or
// tower), `# device` (cuda or cpu, the threads and the card the caller names) and `# mmproj`. `# build_record` lines
// copy the binary's .build record (tools/ref/clefvis/build-clefvis.sh): the tree, its CUDA architectures, compiler and
// CMake options. Every digest is sha256_file's; a file that cannot be read stops the run by name.
// One binary for the card set and its CPU twin: --cpu keeps every layer and the projector on the CPU, and the run
// refuses a card in view (llama.cpp would still move a large product to it), so the twin runs with CUDA_VISIBLE_DEVICES
// empty. Without --cpu the run refuses a backend with no GPU device (GB10's registers as an integrated GPU).
//
// Build: tools/ref/clefvis/build-clefvis.sh (BLOOMERY_MODEL=gemma-embedding2)   Run: tools/ref/emb2/emb2.sh

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define DUMP_TOOL "dump_emb2"
#include "../clefvis/set_writer.h"

struct node_rows {
    int64_t            ne[4];
    std::vector<float> data;
};

struct taps {
    const char * const *             names;  // exact names, or a prefix ending in '-' for a per-layer node
    std::map<std::string, node_rows> got;
    std::string                      error;
    bool                             dense_head = false;
};

static const char * const TEXT_TAPS[]  = { "inp_scaled", "inp_per_layer", "l_out-", "result_norm", "result_embd_pooled", nullptr };
static const char * const TEXT_V1_TAPS[] = { "inp_scaled", "l_out-", "result_norm", "result_embd_pooled", nullptr };
static const char * const TOWER_TAPS[] = { "inp_raw_scaled", "pos_embd", "layer_out-", "pooled", "std_scaled",
                                           "projected", nullptr };

// A per-layer name matches `<prefix><digits>` only: ggml names a view of a node `<name> (reshaped)` and the like.
static bool wanted(const taps & t, const char * name) {
    for (const char * const * n = t.names; *n; ++n) {
        const size_t len = strlen(*n);
        if ((*n)[len - 1] != '-') {
            if (strcmp(name, *n) == 0) return true;
            continue;
        }
        if (strncmp(name, *n, len) != 0 || name[len] == '\0') continue;
        const char * d = name + len;
        while (*d >= '0' && *d <= '9') ++d;
        if (*d == '\0') return true;
    }
    return false;
}

static bool on_node(struct ggml_tensor * t, bool ask, void * user) {
    auto * s = static_cast<taps *>(user);
    if (!wanted(*s, t->name)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        s->error = std::string(t->name) + " is not a contiguous f32 tensor";
        return true;
    }
    // Pooling NONE aliases the normalized rows; the dense head names its own output the same way.
    const char * name = s->dense_head && strcmp(t->name, "result_embd_pooled") == 0 && t->op != GGML_OP_MUL_MAT
        ? "result_norm" : t->name;
    node_rows & r = s->got[name];
    if (!r.data.empty()) {
        s->error = std::string(t->name) + " ran twice in one pass";
        return true;
    }
    for (int i = 0; i < 4; ++i) r.ne[i] = t->ne[i];
    r.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, r.data.data(), 0, ggml_nbytes(t));
    return true;
}

static std::vector<llama_token> read_ids(const std::string & path) {
    std::vector<uint8_t> raw;
    if (!read_file(path, raw)) DIE("cannot read %s", path.c_str());
    std::vector<llama_token> ids = parse_ids(path, raw);
    if (ids.empty()) DIE("%s holds no ids", path.c_str());
    return ids;
}

static std::string max_abs_diff(const float * a, const float * b, size_t n) {
    if (memcmp(a, b, n * sizeof(float)) == 0) return "equal";
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    char buf[64];
    snprintf(buf, sizeof(buf), "differs max_abs %.6g", m);
    return buf;
}

static llama_context * make_ctx(llama_model * model, int n, enum llama_pooling_type pooling, int threads, taps * t) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t) n;
    cp.n_batch      = (uint32_t) n;
    cp.n_ubatch     = (uint32_t) n;
    cp.n_seq_max    = 1;
    cp.embeddings   = true;
    cp.pooling_type = pooling;
    if (t) {
        cp.cb_eval           = on_node;
        cp.cb_eval_user_data = t;
    }
    if (threads > 0) {
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) DIE("cannot make a context of %d positions", n);
    return ctx;
}

static void decode_ids(llama_context * ctx, const std::vector<llama_token> & ids, const std::string & what) {
    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = (int32_t) ids.size();
    const int rc   = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) DIE("%s: llama_decode returned %d", what.c_str(), rc);
}

// The rows of `order` from a tap pass, each named `<prefix>/<row>`: a node the graph did not run, or one it ran that
// `order` does not name, stops the run by name.
static void write_taps(set_writer & out, const taps & t, const std::vector<std::string> & order, const std::string & prefix,
                       const std::map<std::string, std::string> & rename) {
    std::string missing;
    for (const auto & name : order) {
        if (!t.got.count(name)) missing += " " + name;
    }
    if (!missing.empty() || t.got.size() != order.size()) {
        std::string ran;
        for (const auto & [name, r] : t.got) ran += " " + name;
        DIE("%s: the graph ran no node%s; it ran%s", prefix.c_str(), missing.c_str(), ran.c_str());
    }
    for (const auto & name : order) {
        const auto it = rename.find(name);
        const node_rows & r = t.got.at(name);
        out.tensor_f32(prefix + "/" + (it == rename.end() ? name : it->second), "NODE", r.ne, r.data.data());
    }
}

static void run_text(llama_model * model, const std::string & ids_dir, const std::vector<std::string> & cases, int threads,
                     set_writer & out, bool dense_head) {
    const int width = llama_model_n_embd_out(model);
    for (const auto & c : cases) {
        const std::string ids_path = ids_dir + "/" + c + ".ids";
        const std::vector<llama_token> ids = read_ids(ids_path);
        const int n = (int) ids.size();
        out.line("# case\t" + c + "\t" + std::to_string(n) + "\t" + ids_path + "\t" + sha256_file(ids_path));

        llama_context * clean = make_ctx(model, n, LLAMA_POOLING_TYPE_NONE, threads, nullptr);
        decode_ids(clean, ids, c + ": the clean pass");
        std::vector<float> embd((size_t) n * width);
        for (int i = 0; i < n; ++i) {
            const float * e = llama_get_embeddings_ith(clean, i);
            if (!e) DIE("%s: no embedding row for position %d", c.c_str(), i);
            memcpy(&embd[(size_t) i * width], e, width * sizeof(float));
        }
        llama_free(clean);
        const int64_t ne_embd[4] = { width, n, 1, 1 };
        out.tensor_f32(c + "/embd", "MUL_MAT", ne_embd, embd.data());

        llama_context * pooled = make_ctx(model, n, LLAMA_POOLING_TYPE_MEAN, threads, nullptr);
        decode_ids(pooled, ids, c + ": the pooled pass");
        const float * p = llama_get_embeddings_seq(pooled, 0);
        if (!p) DIE("%s: no pooled row", c.c_str());
        const int64_t ne_pooled[4] = { width, 1, 1, 1 };
        out.tensor_f32(c + "/pooled", "MEAN", ne_pooled, p);
        llama_free(pooled);

        taps t{ dense_head ? TEXT_V1_TAPS : TEXT_TAPS, {}, {}, dense_head };
        llama_context * tapped = make_ctx(model, n, LLAMA_POOLING_TYPE_NONE, threads, &t);
        decode_ids(tapped, ids, c + ": the taps pass");
        llama_free(tapped);
        if (!t.error.empty()) DIE("%s: the taps pass: %s", c.c_str(), t.error.c_str());
        std::vector<std::string> order = { "inp_scaled" };
        if (!dense_head) order.push_back("inp_per_layer");
        for (int il = 0; il < llama_model_n_layer(model); ++il) order.push_back("l_out-" + std::to_string(il));
        order.push_back("result_norm");
        order.push_back("result_embd_pooled");
        write_taps(out, t, order, c, { { "result_embd_pooled", "result_embd" } });
        const node_rows & re = t.got.at("result_embd_pooled");
        if (re.data.size() != embd.size()) {
            DIE("%s: result_embd holds %zu values, the clean rows %zu", c.c_str(), re.data.size(), embd.size());
        }
        out.line("# tap_effect\t" + c + "\t" + max_abs_diff(re.data.data(), embd.data(), embd.size()));
        fprintf(stderr, "dump_emb2: %s: %d ids\n", c.c_str(), n);
    }
}

static mtmd_context * make_mtmd(const std::string & mmproj, llama_model * model, bool cpu, int threads, taps * t) {
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu       = !cpu;
    mp.print_timings = false;
    mp.warmup        = false;
    if (threads > 0) mp.n_threads = threads;
    if (t) {
        mp.cb_eval           = on_node;
        mp.cb_eval_user_data = t;
    }
    mtmd_context * c = mtmd_init_from_file(mmproj.c_str(), model, mp);
    if (!c) DIE("cannot load the mmproj %s", mmproj.c_str());
    return c;
}

// The image chunk of a prompt that is the marker alone, encoded; its token count.
static size_t encode_image(mtmd_context * ctx, const std::string & png) {
    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(ctx, png.c_str(), false, mtmd_helper_init_opt_default());
    if (!w.bitmap) DIE("cannot decode %s", png.c_str());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char *        marker = mtmd_default_marker();
    mtmd_input_text     text{ marker, strlen(marker), false, true };
    const mtmd_bitmap * bm     = w.bitmap;
    if (mtmd_tokenize(ctx, chunks, &text, &bm, 1) != 0) DIE("mtmd_tokenize refused %s", png.c_str());
    const mtmd_input_chunk * image = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) image = ch;
    }
    if (!image) DIE("%s tokenized to no image chunk", png.c_str());
    if (mtmd_encode_chunk(ctx, image) != 0) DIE("mtmd_encode_chunk failed on %s", png.c_str());
    const size_t n = mtmd_input_chunk_get_n_tokens(image);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(w.bitmap);
    return n;
}

static void run_tower(llama_model * model, const std::string & mmproj, const std::vector<std::pair<std::string, std::string>> & images,
                      bool cpu, int threads, set_writer & out) {
    const int width = llama_model_n_embd_inp(model);
    taps t{ TOWER_TAPS, {}, {} };
    mtmd_context * clean  = make_mtmd(mmproj, model, cpu, threads, nullptr);
    mtmd_context * tapped = make_mtmd(mmproj, model, cpu, threads, &t);
    for (const auto & [name, png] : images) {
        const size_t n = encode_image(clean, png);
        const float * e = mtmd_get_output_embd(clean);
        if (!e) DIE("%s: no output embeddings", name.c_str());
        std::vector<float> embd(e, e + n * width);
        const int64_t ne_embd[4] = { width, (int64_t) n, 1, 1 };
        out.tensor_f32(name + "/embd", "MUL_MAT", ne_embd, embd.data());

        t.got.clear();
        const size_t n_tap = encode_image(tapped, png);
        if (!t.error.empty()) DIE("%s: the taps encode: %s", name.c_str(), t.error.c_str());
        if (n_tap != n) DIE("%s: the taps encode made %zu tokens, the clean one %zu", name.c_str(), n_tap, n);
        std::vector<std::string> order = { "inp_raw_scaled", "pos_embd" };
        int layers = 0;
        while (t.got.count("layer_out-" + std::to_string(layers))) order.push_back("layer_out-" + std::to_string(layers++));
        if (layers == 0) DIE("%s: the tower ran no layer_out-<il> node", name.c_str());
        order.push_back("pooled");
        // gemma4v.cpp builds `std_scaled` only for a projector that carries std_bias and std_scale.
        const bool std_scaled = t.got.count("std_scaled") > 0;
        if (std_scaled) order.push_back("std_scaled");
        order.push_back("projected");
        write_taps(out, t, order, name, {});
        const node_rows & raw = t.got.at("inp_raw_scaled");
        const node_rows & pr  = t.got.at("projected");
        out.line("# image\t" + name + "\t" + png + "\t" + sha256_file(png) + "\tcanvas " + std::to_string(raw.ne[0]) + "x" +
                 std::to_string(raw.ne[1]) + "\ttokens " + std::to_string(n) + "\tlayers " + std::to_string(layers) +
                 "\tstd_scaled " + (std_scaled ? "yes" : "no"));
        if (pr.data.size() != embd.size()) {
            DIE("%s: projected holds %zu values, the clean embd %zu", name.c_str(), pr.data.size(), embd.size());
        }
        out.line("# tap_effect\t" + name + "\t" + max_abs_diff(pr.data.data(), embd.data(), embd.size()));
        fprintf(stderr, "dump_emb2: %s: %zu tokens\n", name.c_str(), n);
    }
    mtmd_free(tapped);
    mtmd_free(clean);
}

static bool has_card() {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) || ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

int main(int argc, char ** argv) {
    if (argc < 2 || (strcmp(argv[1], "text") != 0 && strcmp(argv[1], "tower") != 0)) {
        DIE("usage: dump_emb2 text|tower ... (see the head of tools/ref/emb2/dump_emb2.cpp)");
    }
    const bool  text_mode = strcmp(argv[1], "text") == 0;
    std::string model_path, mmproj, ids_dir, out_dir, card, record, flags;
    std::vector<std::string> cases;
    std::vector<std::pair<std::string, std::string>> images;
    int  threads = 0;
    bool cpu     = false;
    for (int i = 1; i < argc; ++i) flags += (i > 1 ? " " : "") + std::string(argv[i]);
    for (int i = 2; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (strcmp(argv[i], "-m") == 0 && more) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mmproj") == 0 && more && !text_mode) {
            mmproj = argv[++i];
        } else if (strcmp(argv[i], "--ids-dir") == 0 && more && text_mode) {
            ids_dir = argv[++i];
        } else if (strcmp(argv[i], "--case") == 0 && more && text_mode) {
            cases.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--image") == 0 && more && !text_mode) {
            const std::string v  = argv[++i];
            const size_t      eq = v.find('=');
            if (eq == std::string::npos || eq == 0) DIE("--image takes <name>=<png>, got '%s'", v.c_str());
            images.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        } else if (strcmp(argv[i], "--out") == 0 && more) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--card") == 0 && more) {
            card = argv[++i];
        } else if (strcmp(argv[i], "--build-record") == 0 && more) {
            record = argv[++i];
        } else if (strcmp(argv[i], "--cpu") == 0) {
            cpu = true;
        } else if (strcmp(argv[i], "-t") == 0 && more) {
            const char * arg = argv[++i];
            char * end = nullptr;
            errno = 0;
            const long value = strtol(arg, &end, 10);
            if (errno != 0 || end == arg || *end != '\0' || value < 0 || value > INT_MAX) {
                DIE("-t requires an integer from 0 to INT_MAX, got '%s'", arg);
            }
            threads = static_cast<int>(value);
        } else {
            DIE("unknown or incomplete argument '%s'", argv[i]);
        }
    }
    if (model_path.empty() || out_dir.empty() || card.empty() || record.empty() ||
        (text_mode ? ids_dir.empty() || cases.empty() : mmproj.empty() || images.empty())) {
        DIE("usage: dump_emb2 text|tower ... (see the head of tools/ref/emb2/dump_emb2.cpp)");
    }
    std::ifstream rec(record);
    if (!rec) DIE("cannot read the build record %s", record.c_str());
    std::vector<std::string> record_lines;
    for (std::string l; std::getline(rec, l);) {
        if (!l.empty()) record_lines.push_back(l);
    }

    llama_backend_init();
    if (cpu && has_card()) DIE("--cpu with a card in view: run the CPU twin with CUDA_VISIBLE_DEVICES empty");
    if (!cpu && !has_card()) DIE("no GPU device in view and no --cpu: the card set needs the card");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = cpu ? 0 : 99;
    llama_model * model   = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) DIE("failed to load %s", model_path.c_str());
    const std::string arch = gguf_arch(model_path);
    if (arch != "gemma-embedding2" && arch != "gemma-embedding")
        DIE("%s is a %s file, not an EmbeddingGemma encoder", model_path.c_str(), arch.c_str());
    if (!text_mode && arch != "gemma-embedding2") DIE("%s has no image tower", arch.c_str());

    set_writer out(out_dir);
    const char *      build = getenv("BLOOMERY_REF_BUILD");
    const std::string set_model = text_mode ? model_path : mmproj;
    const size_t      slash     = set_model.rfind('/');
    out.line(std::string("# dump_emb2 ") + (text_mode ? "text" : "tower") + ": llama.cpp mainline EmbeddingGemma, raw f32, little-endian");
    out.line("# model\t" + set_model);
    if (build && *build) out.line(std::string("# build\t") + build);
    out.line("# arch\t" + (text_mode ? arch : gguf_arch(mmproj)));
    out.line("# model_file\t" + (slash == std::string::npos ? set_model : set_model.substr(slash + 1)));
    out.line(std::string("# emb2\t") + (text_mode ? "text" : "tower"));
    out.line("# flags\t" + flags);
    out.line(std::string("# device\t") + (cpu ? "cpu" : "cuda") + "\tthreads\t" + std::to_string(threads) + "\tcard\t" + card);
    if (text_mode) {
        out.line("# model_sha256\t" + sha256_file(model_path));
    } else {
        out.line("# text_model\t" + model_path + "\tsha256\t" + sha256_file(model_path));
        out.line("# mmproj\t" + mmproj + "\tsha256\t" + sha256_file(mmproj));
    }
    for (const auto & l : record_lines) out.line("# build_record\t" + l);

    if (text_mode) {
        run_text(model, ids_dir, cases, threads, out, arch == "gemma-embedding");
    } else {
        run_tower(model, mmproj, images, cpu, threads, out);
    }
    out.finish();
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
