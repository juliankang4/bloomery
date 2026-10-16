// Standalone oracle for gemma4a's host preprocessing. Link the pinned,
// unmodified mtmd-audio.cpp; no model, decoder or device is opened.
#include "gguf.h"
#include "mtmd-audio.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define DUMP_TOOL "dump_mel"
#include "../clefvis/set_writer.h"

// The parameter provider replaces only clip's model loader in this host harness.
const clip_hparams * clip_get_hparams(const clip_ctx *) {
    static const clip_hparams hp = [] {
        clip_hparams h{};
        h.audio_sample_rate = 16000;
        h.audio_n_fft = 512;
        h.audio_window_len = 320;
        h.audio_hop_len = 160;
        h.n_mel_bins = 128;
        return h;
    }();
    return &hp;
}

int main(int argc, char ** argv) {
    if (argc != 2) DIE("usage: dump_mel <new-set-directory>");
    const char * build = getenv("BLOOMERY_REF_BUILD");
    if (!build || !*build) DIE("BLOOMERY_REF_BUILD is required");
    set_writer out(argv[1]);
    out.line("# model\tsynthetic-gemma4a-pcm");
    out.line("# arch\tgemma4a");
    out.line(std::string("# build\t") + build);
    out.line("# arithmetic\tmtmd-audio.cpp\tffp-contract=off");
    mtmd_audio_preprocessor_gemma4a prep(nullptr);
    prep.initialize();
    struct audio_case { const char * name; size_t samples; bool silence; };
    const audio_case cases[] = {
        {"minimum", 2, false}, {"short", 159, false}, {"half-window", 160, false},
        {"frame-one", 161, false}, {"frame-edge", 320, false}, {"frame-two", 321, false},
        {"token-one", 800, false}, {"token-two", 801, false},
        {"sine", 16000, false}, {"silence", 32000, true},
        {"before-30s", 479999, false}, {"at-30s", 480000, false}, {"after-30s", 480321, false},
    };
    for (const auto & c : cases) {
        std::vector<float> pcm(c.samples, 0.0f);
        uint32_t rng = 12345;
        for (size_t i = 0; i < pcm.size() && !c.silence; ++i) {
            rng = 1664525 * rng + 1013904223;
            pcm[i] = 0.4f * sinf(2.0f * float(M_PI) * 440.0f * i / 16000.0f)
                   + float(int32_t(rng >> 16) - 32768) / 131072.0f;
        }
        const std::string name(c.name);
        const int64_t ne_pcm[4] = {int64_t(pcm.size()), 1, 1, 1};
        out.tensor_f32(name + "/pcm", "NONE", ne_pcm, pcm.data());
        std::vector<mtmd_audio_mel> chunks;
        if (!prep.preprocess(pcm.data(), pcm.size(), chunks)) DIE("%s: preprocess failed", c.name);
        for (size_t i = 0; i < chunks.size(); ++i) {
            const auto & chunk = chunks[i];
            const std::string prefix = name + "/" + std::to_string(i);
            const int64_t ne[4] = {chunk.n_len, chunk.n_mel, 1, 1};
            out.tensor_f32(prefix + "/mel", "NONE", ne, chunk.data.data());
            const int64_t scalar[4] = {1, 1, 1, 1};
            const float tokens = float(((chunk.n_len + 1) / 2 + 1) / 2);
            out.tensor_f32(prefix + "/tokens", "NONE", scalar, &tokens);
        }
        fprintf(stderr, "dump_mel: %s samples=%zu chunks=%zu\n", c.name, pcm.size(), chunks.size());
    }
    out.finish();
}
