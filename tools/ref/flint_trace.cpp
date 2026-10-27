// Flint forced-ID diagnostic trajectory, not a reference-set family or accuracy gate.
// Only the last input requests logits: the prompt's last layer is therefore already cropped.
// Node taps change scheduler segmentation; --no-taps is the same-engine logits-only control.
// Model SHA256 and binary/source identity belong in the external execution receipt.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ref_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace fs = std::filesystem;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static std::ofstream output(const fs::path & path, bool binary = false) {
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode(0)));
    return stream;
}

static void write_row(std::ofstream & out, const float * row, size_t width) {
    out.write(reinterpret_cast<const char *>(row), static_cast<std::streamsize>(width * sizeof(float)));
}

static std::string metadata(llama_model * model, const char * key) {
    char value[256];
    const int n = llama_model_meta_val_str(model, key, value, sizeof(value));
    require(n >= 0 && static_cast<size_t>(n) < sizeof(value), std::string("missing/oversized model metadata: ") + key);
    return value;
}

static std::string actual_path(const std::string & path) {
    const std::string result = fs::canonical(path).string();
    require(result.find_first_of("\t\r\n") == std::string::npos, "path contains a TSV separator: " + path);
    return result;
}

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
};

struct batch_guard {
    llama_batch batch = llama_batch_init(32, 0, 1);
    ~batch_guard() { llama_batch_free(batch); }
};

struct tap_state {
    int width;
    int layers;
    int call = -1;
    int call_tokens = 0;
    bool active = false;
    std::vector<std::string> names;
    std::vector<unsigned char> seen;
    std::vector<float> rows;
    std::vector<uint8_t> staging;
    std::ofstream * tensors;
    std::string error;

    tap_state(int width_, int layers_, std::ofstream * tensors_) :
        width(width_), layers(layers_), seen(layers_ + 1), rows(static_cast<size_t>(layers_ + 1) * width_),
        tensors(tensors_) {
        for (int i = 0; i < layers; ++i) names.push_back("l_out-" + std::to_string(i));
        names.push_back("result_norm");
        staging.reserve(static_cast<size_t>(width) * sizeof(float));
    }

    int selected(const char * name) const {
        const auto it = std::find(names.begin(), names.end(), name);
        return it == names.end() ? -1 : static_cast<int>(it - names.begin());
    }

    void begin(int call_, int tokens_) {
        call = call_;
        call_tokens = tokens_;
        active = true;
        std::fill(seen.begin(), seen.end(), 0);
        error.clear();
    }

    void capture(const ggml_tensor * t, int index) {
        const std::string label = "call " + std::to_string(call) + " tensor " + t->name;
        require(!seen[index], label + ": duplicate capture");
        const int expected_rows = index < layers - 1 ? call_tokens : 1;
        require(t->ne[0] == width && t->ne[1] == expected_rows && t->ne[2] == 1 && t->ne[3] == 1,
                label + ": invalid shape (expected [" + std::to_string(width) + "," +
                std::to_string(expected_rows) + ",1,1])");
        size_t element_bytes;
        switch (t->type) {
            case GGML_TYPE_F32: element_bytes = sizeof(float); break;
            case GGML_TYPE_F16: element_bytes = sizeof(ggml_fp16_t); break;
            case GGML_TYPE_BF16: element_bytes = sizeof(ggml_bf16_t); break;
            default: throw std::runtime_error(label + ": unsupported type " + ggml_type_name(t->type));
        }
        require(t->buffer && t->data, label + ": unallocated tensor");
        require(t->nb[0] >= element_bytes, label + ": invalid element stride");
        const size_t selected_row = static_cast<size_t>(t->ne[1] - 1);
        const size_t bytes = ggml_nbytes(t);
        require(selected_row == 0 || t->nb[1] <= bytes / selected_row, label + ": invalid row stride");
        const size_t offset = selected_row * t->nb[1];
        require(offset <= bytes && element_bytes <= bytes - offset, label + ": row outside tensor");
        require(static_cast<size_t>(width - 1) <= (bytes - offset - element_bytes) / t->nb[0],
                label + ": element strides outside tensor");
        const size_t span = static_cast<size_t>(width - 1) * t->nb[0] + element_bytes;
        staging.resize(span);
        // Copy while the scheduler still owns this live output, including view strides.
        ggml_backend_tensor_get(t, staging.data(), offset, span);
        float * dest = rows.data() + static_cast<size_t>(index) * width;
        for (int i = 0; i < width; ++i) {
            const uint8_t * src = staging.data() + static_cast<size_t>(i) * t->nb[0];
            float value;
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(&value, src, sizeof(value));
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_t half;
                std::memcpy(&half, src, sizeof(half));
                value = ggml_fp16_to_fp32(half);
            } else {
                ggml_bf16_t half;
                std::memcpy(&half, src, sizeof(half));
                value = ggml_bf16_to_fp32(half);
            }
            if (!std::isfinite(value)) {
                throw std::runtime_error(label + ": nonfinite element " + std::to_string(i));
            }
            dest[i] = value;
        }
        *tensors << call << '\t' << t->name << '\t' << ggml_type_name(t->type);
        for (int i = 0; i < 4; ++i) *tensors << '\t' << t->ne[i];
        for (int i = 0; i < 4; ++i) *tensors << '\t' << t->nb[i];
        *tensors << '\t' << bytes << '\t' << selected_row << '\t' << ggml_op_name(t->op)
                 << '\t' << (ggml_is_contiguous(t) ? 1 : 0) << '\n';
        seen[index] = 1;
    }

    void finish() {
        active = false;
        require(error.empty(), error);
        for (int i = 0; i <= layers; ++i) {
            require(seen[i], "call " + std::to_string(call) + " tensor " + names[i] + ": missing capture");
        }
    }
};

static bool on_node(ggml_tensor * tensor, bool ask, void * user) {
    auto & state = *static_cast<tap_state *>(user);
    if (!state.active) return !ask;
    const int index = state.selected(tensor->name);
    if (ask) return index >= 0;
    // Never unwind a C callback or cancel a partially computed graph as if it succeeded.
    if (index >= 0 && state.error.empty()) {
        try {
            state.capture(tensor, index);
        } catch (const std::exception & e) {
            state.error = e.what();
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    try {
        const char * usage = "usage: flint_trace -m MODEL --ids FILE --prompt 22 --out NEWDIR --ctx 1024 [--no-taps]\n";
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                std::fputs(usage, stdout);
                std::fputs("Read exactly 96 whitespace-separated decimal token IDs; never retokenize or stop on EOG.\n"
                           "Matches HTTP: 21-ID prefix prefill, last prompt ID as next, then 74 forced steps.\n"
                           "Default: CPU layer/norm node taps and full logits. --no-taps: full logits only.\n"
                           "NEWDIR must not exist; MANIFEST.tsv is installed only after complete capture.\n", stdout);
                return 0;
            }
        }
        std::string model_arg, ids_arg, out_arg;
        int prompt = 22, capacity = 1024;
        bool taps = true;
        for (int i = 1; i < argc; ++i) {
            const bool more = i + 1 < argc;
            if (std::strcmp(argv[i], "-m") == 0 && more) model_arg = argv[++i];
            else if (std::strcmp(argv[i], "--ids") == 0 && more) ids_arg = argv[++i];
            else if (std::strcmp(argv[i], "--out") == 0 && more) out_arg = argv[++i];
            else if (std::strcmp(argv[i], "--prompt") == 0 && more) {
                if (!parse_i32("flint_trace", "--prompt", argv[++i], 1, prompt)) return 2;
            } else if (std::strcmp(argv[i], "--ctx") == 0 && more) {
                if (!parse_i32("flint_trace", "--ctx", argv[++i], 1, capacity)) return 2;
            } else if (std::strcmp(argv[i], "--no-taps") == 0) taps = false;
            else throw std::runtime_error(std::string("unknown/incomplete argument: ") + argv[i]);
        }
        require(!model_arg.empty() && !ids_arg.empty() && !out_arg.empty(), usage);
        require(prompt == 22 && capacity == 1024, "flint-trajectory-v2 requires --prompt 22 --ctx 1024");
        const uint32_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4 &&
                std::numeric_limits<float>::is_iec559, "requires little-endian IEEE754 f32 host");
        const std::string model_path = actual_path(model_arg);
        const std::string ids_path = actual_path(ids_arg);
        require(out_arg.find_first_of("\t\r\n") == std::string::npos, "output path contains a TSV separator");
        struct stat model_stat {};
        require(stat(model_path.c_str(), &model_stat) == 0 && S_ISREG(model_stat.st_mode) &&
                model_stat.st_size == 25054688704LL, "model file is not the complete Flint UD-Q4_K_XL release size");

        std::vector<llama_token> ids;
        ids.reserve(96);
        std::ifstream source(ids_path);
        require(bool(source), "cannot read IDs");
        std::string record;
        while (source >> record) {
            int32_t id;
            require(!record.empty() && record[0] >= '0' && record[0] <= '9' && parse_token_id(record, id),
                    "invalid decimal token ID at position " + std::to_string(ids.size()) + ": " + record);
            require(ids.size() < 96, "ID file must contain exactly 96 IDs: extra token at position 96");
            ids.push_back(id);
        }
        require(source.eof() && !source.bad(), "cannot finish reading IDs");
        require(ids.size() == 96, "ID file must contain exactly 96 IDs, found " + std::to_string(ids.size()));

        backend_guard backend;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
        require(bool(model), "cannot load model");
        const int width = llama_model_n_embd(model.get());
        const int layers = llama_model_n_layer(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::string arch = metadata(model.get(), "general.architecture");
        const std::string model_name = metadata(model.get(), "general.name");
        require(arch == "flint" && model_name == "Flint Preview" && width == 2048 && layers == 49 && vocab == 150144 &&
                metadata(model.get(), "general.file_type") == "15", "model metadata is not the Flint Preview UD-Q4_K_XL release");
        for (size_t i = 0; i < ids.size(); ++i) {
            require(ids[i] >= 0 && ids[i] < vocab, "token outside model vocabulary at position " + std::to_string(i));
        }
        const int calls = 1 + static_cast<int>(ids.size()) - prompt;
        require(fs::create_directory(out_arg), "output directory must be new: " + out_arg);
        const fs::path out = fs::canonical(out_arg);
        auto manifest = output(out / "MANIFEST.tsv.partial");
        auto steps = output(out / "steps.tsv");
        auto tokens = output(out / "tokens.ids");
        auto logits_out = output(out / "logits.f32", true);
        auto tensors = output(out / "tensors.tsv");
        std::ofstream layers_out, norm_out;
        if (taps) {
            layers_out = output(out / "layers.f32", true);
            norm_out = output(out / "norm.f32", true);
        }
        manifest << "protocol\tflint-trajectory-v2\nmodel\t" << model_path << "\narch\t" << arch
                 << "\nmodel_name\t" << model_name << "\nmodel_bytes\t" << model_stat.st_size
                 << "\nmodel_identity_validation\tmetadata-and-file-size;SHA256-in-external-receipt\n"
                 << "mode\tcpu\ninstrumentation\t" << (taps ? "node-taps" : "no-taps")
                 << "\nids_source\t" << ids_path << "\nwidth\t" << width << "\nlayers\t" << layers
                 << "\nvocab\t" << vocab << "\nprompt\t" << prompt << "\ntotal_ids\t" << ids.size()
                 << "\ncalls\t" << calls << "\nlogits\tlogits.f32\nlayers_file\t" << (taps ? "layers.f32" : "absent:not-requested")
                 << "\nprompt_submission\tprefix-prefill-then-next\nprefix_prefill_ids\t21\nfirst_observed_call\tone-id-next"
                 << "\nnorm_file\t" << (taps ? "norm.f32" : "absent:not-requested")
                 << "\nlogits_layout\t[call][vocab];little-endian-f32\nlayers_layout\t[call][layer][width];little-endian-f32"
                 << "\nnorm_layout\t[call][width];little-endian-f32\ntensor_metadata\ttensors.tsv"
                 << "\nngpu\t0\nload_mode\tmmap\nuse_extra_bufts\tfalse\nthreads\t4\nthreads_batch\t4"
                 << "\noffload_kqv\tfalse\nop_offload\tfalse\nswa_full\tfalse\nkv_unified\tfalse"
                 << "\ntype_k\tf16\ntype_v\tf16\nflash_attn\tauto\nno_perf\ttrue\n";
        manifest.flush();
        tensors << "call\tname\ttype\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tbytes\tselected_row\top\tcontiguous\n";
        tap_state state(width, layers, &tensors);
        auto cp = llama_context_default_params();
        cp.n_ctx = capacity;
        cp.n_seq_max = 1;
        cp.n_batch = cp.n_ubatch = 32;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.no_perf = true;
        cp.swa_full = false;
        cp.kv_unified = false;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        if (taps) {
            cp.cb_eval = on_node;
            cp.cb_eval_user_data = &state;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "cannot create CPU context");
        require(llama_n_ctx_seq(ctx.get()) == static_cast<uint32_t>(capacity) && llama_n_seq_max(ctx.get()) == 1 &&
                llama_n_batch(ctx.get()) == 32 && llama_n_ubatch(ctx.get()) == 32, "actual context/batch capacities differ from requested protocol");
        manifest << "slots\t" << llama_n_seq_max(ctx.get()) << "\nctx_per_slot\t" << llama_n_ctx_seq(ctx.get())
                 << "\nctx_total\t" << llama_n_ctx(ctx.get()) << "\nn_batch\t" << llama_n_batch(ctx.get())
                 << "\nn_ubatch\t" << llama_n_ubatch(ctx.get()) << '\n';
        manifest.flush();
        for (llama_token id : ids) tokens << id << '\n';
        tokens.close();
        steps << "call\tprocessed_pos\tpredict_pos\tinput_id\targmax\n";
        batch_guard batch;
        // Match HTTP Slot::close_call: the prefix has no observation; the last prompt ID is next.
        for (int call = -1; call < calls; ++call) {
            const int at = call < 0 ? 0 : prompt - 1 + call;
            const int count = call < 0 ? prompt - 1 : 1;
            auto & b = batch.batch;
            b.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                b.token[i] = ids[at + i];
                b.pos[i] = at + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = i == count - 1;
            }
            if (taps && call >= 0) state.begin(call, count);
            const int status = llama_decode(ctx.get(), b);
            if (taps && call >= 0) state.finish();
            require(status == 0, "llama_decode failed at call " + std::to_string(call) + ": " + std::to_string(status));
            if (call < 0) continue;
            const float * row = llama_get_logits_ith(ctx.get(), -1);
            require(row != nullptr, "missing logits at call " + std::to_string(call));
            int argmax = 0;
            for (int i = 0; i < vocab; ++i) {
                if (!std::isfinite(row[i])) {
                    throw std::runtime_error("nonfinite logits at call " + std::to_string(call) + " lane " + std::to_string(i));
                }
                if (row[i] > row[argmax]) argmax = i;
            }
            write_row(logits_out, row, vocab);
            if (taps) {
                write_row(layers_out, state.rows.data(), static_cast<size_t>(layers) * width);
                write_row(norm_out, state.rows.data() + static_cast<size_t>(layers) * width, width);
            }
            const int processed = at + count - 1;
            steps << call << '\t' << processed << '\t' << processed + 1 << '\t' << ids[processed] << '\t' << argmax << '\n';
        }
        logits_out.close();
        steps.close();
        tensors.close();
        if (taps) {
            layers_out.close();
            norm_out.close();
        }
        manifest << "complete\t1\n";
        manifest.close();
        fs::rename(out / "MANIFEST.tsv.partial", out / "MANIFEST.tsv");
        std::fprintf(stderr, "flint_trace: %d calls, width=%d layers=%d vocab=%d taps=%s into %s\n",
                     calls, width, layers, vocab, taps ? "on" : "off", out.c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "flint_trace: %s\n", e.what());
        return 1;
    }
}
