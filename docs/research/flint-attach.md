# Flint: reader and common-owner attachment design (#632)

Design snapshot: **`e53cabebbb9b16249c81b7c26a026da1670f7d9d`**, requested by
[#632](https://github.com/midagedev/bloomery/issues/632). The source-gap observations below describe
that snapshot. The integration adds an actual `gpu-flint` body, application session and shared
HTTP seat, not just architecture admission. Sections 10–11 retain source-specific historical
execution evidence; section 12 records the current existing-PR cutover and remaining acceptance.

## Six-slot handoff closure checks — 2026-10-26

The existing #781 branch was ordinarily merged with main
`15778e2cb2dd7af428d2dd14259e6b0ea8ca6b89`; its published history was not rewritten.
On source `9bfa55adc257f7e6aae4952aca06246eddaeef70`, the owning DS41 library ran on
Hikari's RTX 3090 (GPU0): **38 passed, 0 failed, 0 ignored**, in 28.75 seconds.
`hw_f32_tier_handoffs_conserve_consumed_routes` exercised four, six and eight routes,
including partial widths, invalid IDs, poisoned unconsumed rows and changed-input graph replay.
The following host-only changes format the guard and replace a redundant slot-dependent
error label with the common `F32 tier handoff` context. The measured common-name census
falls from 39 to 37; the ABI-stable generated kernel names and dispatch stay unchanged.
The earlier GPU run is not relabeled as a run of these later commits.

The actual `generate_ds41` carrier at `b9b6e013ebfab5748688be52e9a5d398ac2fc9a3`
subsequently measured `ffn_handoff_6_tier` on sm_86 / physical cc8.6:
ptxas13.3.33 spill0, driver-JIT local0, both register counts26 (driver595.91.07,
JIT CUDA13.2, Rust1.99.1 / cuda-oxide9df9e7ce / repository patch31ab6a1d3f58).
Only that measured row is added. Its normalized entry digest is
`a176a9a5846778e40ba99fc2be637d90` (decl1, 163 lines).
Measurements establish only physical cc8.6. The other three
sm_86/sm_121/sm_121a + physical cc12.1 rows remain unmeasured: they carry the entry at the cc8.6
values under a `PIN(2026-10-27)` until a GB10 scan runs.

## Host reader status

`arch/flint` reads independent bool SWA/RoPE masks, explicit head widths, the dense lead,
sigmoid router facts, optional selection bias and one explicitly sized ungated shared MLP.
It reuses the common metadata validators, tensor stems, classification and dimension checks.
`arch::per_layer_bool` owns exact-length bool-array validation for both Flint masks and
GemmaEmbedding2's sliding mask: missing keys, scalars, wrong lengths and non-bool items are refused.
Present `expert_weights_scale` uses `arch::finite_positive`; only an absent key defaults to 1,
with that default recorded.
The common needs walk emits q/k normalization even on NoPE layers. `program_of(Flint)` selects
`FlintBody`, whose coverage names the operations its loader actually binds: head128/group8
sink attention, independent NoPE/NeoX and SWA masks, and the real floor router.
`Router.norm` and `RouterRule.norm` keep the old selected-sum rule distinct from Flint's floor.
The activated-value protection and explicit residual association are loaded FFN policies.

Fault formatting uses `bloomery_device::fault`'s shared layout/decoder and the
`gpu::arch::step_order` registry. The body delegates fault checks and captured execution to the
common chain runtime; the architecture registry's neutral order is not numerical evidence.

Run `BLOOMERY_BOX_ENV="BLOOMERY_MODEL=/absolute/path/flint-header.gguf" BLOOMERY_BOX_LOCAL=1 just gate-flint-meta`.
Set `BLOOMERY_ENV_FILE` to the local toolchain environment file if it is not `~/bloomery-env.sh`.
Keep the outer `BLOOMERY_MODEL` unset or a tool-profile name: `box.sh` reads it before launching the command.
The model-path assignment belongs in `BLOOMERY_BOX_ENV`, which reaches the test after profile resolution;
that channel accepts paths without whitespace.
The gate hashes the first 5,470,528 bytes before using `gguf::inventory_of`, `write::Layout`
and `Split::header_only`; it reads no tensor payload and never treats a sparse file as real weights.
The pinned header yielded 779 tensors, 49 layers, 2 dense/47 routed blocks, 49 head-normalization
needs, 35 rotary needs and 49 sink-attention needs. An external reader smoke also exercised
short/malformed masks, zero window, nonfinite scale, missing sink/norm/shared tensors, a malformed
optional bias and an unknown tensor: each was refused with the offending key/name.

The independent common Jinja and tokenizer changes are tracked by #632; a reader does not imply
those changes are present in an older checkout. Fixture/refset registration is intentionally absent:
the artifact license/provenance conflict below is unresolved, and runtime support still requires the
remaining acceptance rows. This header gate is not an end-to-end model gate.

### Runtime handoff and host policy

The design, common template support, tokenizer/four-EOG contract and reader (#676–#680) are merged.
The pinned-header consumer now also checks actual Program coverage and planning on the complete
local GGUF: 779 tensors, 49 layers, 2 dense/47 routed, 49 head norms, 35 rotations and 49 sinks.
This reads the header, not tensor contents; it does not prove CUDA execution.

The maintainer [assigned the model port and offered card runs](https://github.com/midagedev/bloomery/issues/632#issuecomment-6352066886).
The later public QK-RMS, sink/SWA and sigmoid train supplies reusable primitives. The new common
`attention_h128` binding combines those with actual ring append/attention kernels; it is not an
identity-RoPE substitute or a padded-head adaptation of another family.

The common `route_body::sigmoid_floor_256_6` binding now supplies Flint's fixed
`max(sum_selected, 2^-14)` rule. `RouterKernels::enqueue_router` and `enqueue_router_rows`
take `Option<&DeviceBuffer<f32>>` for a real optional selection-only bias. The common
`RouterEntry`/`RouteIn` consuming paths carry that option; `RouteBlock` allocates no zero-bias
buffer for this binding. Absent bias borrows the existing router-weight slice only as an unused
device ABI argument; no bias data is fabricated, copied or read. The floor policy is a compile-time
specialization of the same score, publication and top-k bodies, not a per-step normalization flag.

Both metadata bindings are `RouterBody::SigmoidFloor` at exactly 256/top6: sigmoid, no shared
sigmoid gate, `SelectedSumFloor`, with or without selection bias. Bias changes only the ranking.
The six selected unbiased scores are summed in f64 slot order, narrowed to f32, divided by the
floored sum and then scaled. Equal selection keys choose larger expert ids. Nonfinite logits,
selection keys or resulting weights raise `FaultSite::Router`; an invalid selection key poisons
all six weights. Existing epsilon-added entries, launch ABIs and legacy norm-argument rules remain
unchanged. A legacy `RouterEntry` given no required bias returns a named state error.

`gate-flint-router` now consumes the actual floor binding through `RouterEntry` for planted and
dense rows, both bias modes, active tiny-sum floors, zero sums, ties, nonfinite logits/bias/scale,
one-token/batch bit equivalence and eager/captured replay. Its independent host weight expression
uses the declared floor and rejects an epsilon-added or biased-weight implementation.
The two entries (`sigmoid_floor_256_6_router` and
`sigmoid_floor_256_6_router_pick`) need same-toolchain add-class PTX/JIT comparisons
and measured resource pins; compiling an entry or running a numerical gate is not
a resource census. No unmeasured pin is supplied.

The loaded `FlintBody`, app session and serve seat now compose these common owners.
Program coverage is present; it must not be confused with whole-model numerical
acceptance. Paging/resident slots, reference provenance, eager/captured execution,
HTTP behavior and affected common/tier consumers retain their own acceptance gates.
Legacy metadata booleans share `RouterNorm::from_legacy_bool`: false keeps scores and true uses the
existing epsilon-added selected sum. Flint's distinct floor mapping stays in its reader.

### Ordinary IQ4_XS shared-site implementation

IQ4_XS now has an ordinary file-layout/dense entry in QROWS and a common site family.
The loader keeps its original block words; `site::kgemv` uses the existing IQ4_XS
entry over one matrix and `site::gemm` uses the existing grouped GEMM. The common
gated block selects a split gate/up and sizes the down's q8_1 scratch from the
actual activation form. Its column walk and the shared wide-prompt owner use
those same sites. IQ4_XS is not relabeled as a routed expert or requantized.

A compiled public `file_site` consumer on the pinned header refused all three
layer42 shared matrices before this change and accepts their IQ4_XS/Q8x128
bindings after it. This reads no payload and proves neither loading nor arithmetic.
`hw_iq4_shared_loads_and_runs_narrow_columns_and_wide_replay`, owned by
`gate-gpu-ds41-lib`, covers real packed-weight loading, unchanged stored words,
the ordinary `SharedExpert` layout, unequal matrix widths, one/eight-column block
execution and the wide-prompt transition/captured replay. Wide scratch holds only
the q8_1 input and down forms these sites read. Its nonnegative constant-weight
case derives a scalar floating-operation bound; uniform codes do not detect nibble
decode ordering. That remains a separate IQ4 kernel-gate obligation, not a
whole-model accuracy pin or a claim made by this wrapper fixture.

The pinned current-main merge (`cb8d92ebddd1ce1a325a15450c06a01f57ecf88c`) also
reconciles its existing Qwen3.8 projected-site consumer: `site::PROJECTED` includes
IQ4_XS, and selected dispatch uses the supplied stream. Projected attention/shared
sites are admitted; the K=640 shared down and unsupported types/shapes remain refused.
This reuses the existing program and kernel, not a new model implementation.
Flint's implemented Program separately binds the real ordinary dense/shared operations.

At component source `b0fc05e5349330927ceb33ce6f30e584f25ecf4d`, the typed GGUF
`iq4_sites_keep_shape_and_launch_admission_separate` consumer passed1 case,
0 failed/ignored,430 filtered. Its successful arm uses `PROJECTED`, not merely
`ALL`; this checks actual file admission rather than copying a metadata registry.
The case is metadata-only and creates no CUDA context; oxide owns its linking.
It is neither fresh NVIDIA stream/launch execution nor a run of this integration.

Pinned upstream `95a483fc242d00dec7987efe4a05e24b5f651c1f` subsequently adds the
split gate/up path that materializes f32 SwiGLU for a down that reads f32.
The obsolete split-to-f32 refusal is removed, not re-pinned; mismatched gate/up
forms still refuse by semantic Shape/caller/type data. IQ4_XS follows the shared
q8x128 predicate in both row dispatch and `writes_h`, so the GLM prompt consumer
chooses the buffer that is actually written instead of allocating/reading a fake f32 row.
At source `5abe1ae66b651446938e334331c06683cef793a4`, the actual block metadata
consumers passed5 cases,0 failed/ignored,34 filtered, including both IQ4 down forms.
These host cases are not execution evidence for the remaining GPU cases.

On 2026-10-26, source `203f5125a3c48a5b700c6f43318cfcb9c77ff169` executed the owning
DS41 library on Hikari GPU0 (RTX3090 cc8.6, driver595.91.07):38 passed,0 failed,0 ignored,
28.61s, including the named IQ4 consumer above. Rust1.99.1 / pinned cuda-oxide9df9e7ce
with repository patch31ab6a1d3f58 and LLVM21.1.8 were used.
Its canonical comparison to main15778e2c covered4bins/8carriers/1657entries: all identical,
non-JIT resource rows equal; no JIT-resource comparison is claimed.
All-target native Clippy/formatting and source203f5125 GitHub T0/lint passed.
These are source-specific historical results, not another hardware execution of
later diagnostic, test or documentation changes. Wider affected real-model,
placement and residency coverage remains explicit; this does not establish a
Flint oracle, physical cc12.1, ROCm or performance.

### Selected expert arms and tier scratch

Common card and tier wrappers resolve `LegArms` from their selected `Program`, rather than assuming
every family has one gate/up type or one down activation form. Their resident layers determine the
union of `Q8Act` and `Q8Blocks32` buffers. Plane layouts remain restricted to the selected program's
arms; Solar's host-only triples do not enlarge its card scratch.

Placement charges a tier's selected down forms jointly with its expert count. The same kernels-owned,
checked sizing arithmetic is used by allocation and planning; malformed or overflowing public scratch
descriptors are refused by name. The plan records the selected reserve, and loaders check that amount
instead of the machine descriptor's upper bound. No always-both allocation or reserve-equality waiver
is used.

The wrapper fixture loads each expert part through the selected program's `routed_rule` format,
`Segment::buffer_bytes` and `Weights::load_rows`, including selected-only IQ3_S. Its mixed loads
compare real step, pass, batch and tier allocations and rows against the common leg. A mismatched
gate/up pair remains refused; Solar's admitted Q6_K down is not a rejection case. Tier activation
selection remains the family's contract, independently of the program-owned stack admission.

The 2026-10-26 cutover removes the obsolete always-Q8 tier-reserve regression:
selected mixed-layer and mismatched-gate/up coverage remains in
`tier_scratch_follows_selected_mixed_layers`, with allocation and budget boundaries
covered by the placement and GPU wrapper consumers. The MiniMax planner test retains
its reserve and idle-tier behavior checks rather than comparing descriptor representation
against a different constructor.
Host-only projections intentionally have no card totals: fixed reserves remain owned by their machine.
Only a dynamic tier-leg descriptor requires the plan's selected card totals. The existing node paging
budget regression exercises this distinction without a GPU, including OS reserve and constrained host room.

The current-main cutover keeps reserve-only draft holders, per-card budgets and each
family's activation contract. `LegTier` receives its activation converter at load;
selected `Program` arms replace the redundant family arm hook. Share-budget floors
use the same selected reserve calculation.

`tier_leg_batch_bytes` and the explicit-form `try_tier_leg_batch_bytes_of` return
`Result<TierBatchBytes, PlacementError>` and construct candidate-selected descriptors.
An empty tier selects no down form, matching `LegTier`'s actual allocation instead of
advertising unused Q8Act scratch. Qwen3/Qwen35, MiMo and Muse use this checked path.
Qwen3/Qwen35 and MiniMax derive their descriptor bounds from the described model's
admitted forms through `accepts::leg_down_forms`, as GLM5Next and Solar do.
The Qwen public wrappers take `PlanInputs`, not geometry alone: a supported
Q8_0, Q5_1 or IQ4_NL candidate must not fail the whole plan just because its
32-value form remains unselected at the tier's budget boundary.
No fallback format can admit a mismatched triple.
The checked constructors and non-leg paths share staging and host byte formulas.
Malformed geometry and overflow are refused rather than panicking; a dense Qwen3.6
request for an expert tier is a named metadata refusal, not zero expert geometry.
Separate tile/route-based fixed-reserve paths remain unchanged.

`TierLegScratch::input_scales` names an optional physical f32 scale per chunk
slot. Its checked byte formula includes `4 * chunk * n_used` when enabled,
including an empty selected-down union; legacy descriptors leave it disabled.
The same descriptor prices the loader's scale allocation without reserve padding.

The planner's selected scratch calculation has one union/byte rule. Fill prepares
per-layer forms once for dynamic tier-leg cards, including multiple forms in one
layer, and reuses them with its cached footprint instead of rescanning tensors for
each candidate. Fixed-reserve cards need no selected-form cache.

The Solar budget regression starts from a typed mixed GGUF description and exercises
the planner's one-byte refusal/success boundary without overwriting its described
prompt reserve. With byte granules (no reusable dense padding), the exact dense-and-prompt
budget leaves no room for expert weights.
The real mixed-wrapper hardware case also checks allocation of a configured empty tier.

Historical component proof at `3071a9f0d8ed4773692e0c040dece222b90f6bec`:
GPU library418 passed plus10 held passed, verbs1 passed, and all1657 canonical entries
identical to main15778e2c (4bins/8carriers, non-JIT rows equal).
The component repair at `c26025b1fb9fad1af399401d27cf709005736ba0` then executed the extended
real CUDA mixed/empty-tier consumer:1 passed,0 failed,0 ignored,427 filtered,24.18s.
Its fresh4bin/8carrier canonical comparison again found all1657entries identical to main15778e2c,
with non-JIT rows equal. Native formatting/Clippy passed (warning3=ratchet3); model557 and
placement202 passed,0 failed,5 model tests ignored,0 filtered.

The historical component contract source `7e37ef9a6ebbd766032ee521a0ee16e25f163851`
executed the real CUDA mixed/empty-tier consumer on RTX3090 cc8.6:1 passed,
0 failed/ignored,427 filtered,23.98s. Its metadata-only
`a_dense_header_plans_through_qwen35_on` frontend consumer also passed1 case,
including the named dense-header expert-tier refusal; that is not a second GPU case.
The review code passed native formatting/all-target Clippy (warning3=ratchet3),
chain23, GGUF67, model557 and placement202 (five model cases ignored,0 filtered).
An exact-source7e37ef9a canonical comparison against main15778e2c found all1657
entries identical across4bins/8carriers and non-JIT rows equal; it did not compare
driver-JIT resource rows. The older full-library counts above remain historical,
not a new full-library run of the review repairs or a run of this integration.

At component merged source `c38d603ca1d33c6c2bede63aba102199715fb9ef`, native formatting
and all-target Clippy passed, as did chain23, GGUF67, model559 and placement202
(five model cases ignored, zero failed/filtered). The actual oxide-linked
`a_dense_header_plans_through_qwen35_on` frontend consumer passed1 case,
0 failed/ignored,6 filtered, including the named expert-tier refusal.
This is metadata-only host execution, not fresh NVIDIA execution or a run of this integration.

The host-only planner covers one-byte boundaries before the first resident layer and another
down form, plus descriptor overflow refusal. Neither this nor the component CUDA proof is a
whole-model oracle or every affected model/residency gate; canonical equality does not waive
changed host/allocation consumers. The integration's body/app/seat consume the selected forms.

### Six-slot F32 tier handoff implementation

`HandoffKernels::enqueue_handoff_tier` now selects a genuine six-slot F32 entry
beside the existing four- and eight-slot entries. Both host and tier layouts must have the
same supported slot count and activation width; compressed tier images remain
refused. The entry uses the existing copy/place/fault helpers and the existing
subsequent go/publication protocol; the integration's configured routed block consumes it.

`hw_f32_tier_handoffs_conserve_consumed_routes`, owned by
`gate-gpu-ds41-lib`, consumes the four-, six- and eight-slot handoffs' weights and emitted
places through the production tier accumulator and adds the host sum. It covers partial
widths, invalid ids, poisoned non-owning rows, trailing sentinels and changed replay
weights. Its host sum is precomputed and activation images are checked by
readback: it is not expert execution or two-card completion/publication proof.
Earlier integrated-source results do not verify this branch after its merge with current
main; the final-source hardware result must be recorded separately.

The new `ffn_handoff_6_tier` entry needs add-class PTX/JIT/resource and
owning-gate evidence; the four-slot, eight-slot and other existing entries must remain
unchanged. The cc12.1 tables pin it at the cc8.6 scan's values until an observed NVIDIA JIT run.
Actual six-slot body/tier and two-card completion remain integration work, not
capabilities inferred from this primitive.
`check-ptx-names` is green with those pinned rows; a GB10 scan replaces them with the
observed spill/JIT-local measurements. Offline no-JIT scans cannot supply that column.




## 1. Sources and observed artifact

- Official [config](https://huggingface.co/RecursiveSI/flint/raw/2a247be0c0aec895ccb0ac7ca92229ec6df37288/config.json),
  [model card](https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/README.md) and
  [chat template](https://huggingface.co/RecursiveSI/flint/raw/2a247be0c0aec895ccb0ac7ca92229ec6df37288/chat_template.jinja):
  RecursiveSI/flint **`2a247be0c0aec895ccb0ac7ca92229ec6df37288`**.
- Actual GGUF specimen: unsloth/flint-GGUF **`20468edfc38ab43d6914cc88b08cbecb05d0fc62`**,
  [flint-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/flint-GGUF/blob/20468edfc38ab43d6914cc88b08cbecb05d0fc62/flint-UD-Q4_K_XL.gguf).
  The Hub lists 25,054,688,704 bytes. Its LFS digest is provenance from the Hub, **not** a locally verified full-file digest.
- Numerical/reference contract: mainline llama.cpp [Flint implementation](https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/models/flint.cpp)
  and [converter](https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/conversion/flint.py),
  merge **`a37cc46104eb50900d5ab4a9ba02a0abdac039db`** of [#30928](https://github.com/ggml-org/llama.cpp/pull/30928).
  Mainline is the oracle; an ik build without Flint is not an alternative oracle.
- [Measured header evidence](flint-attach-evidence.json) records pins, hashes, metadata, shapes and exact
  per-role/per-layer quant types. All shape axes below are GGUF `ne[]`, contiguous axis first.

The existing `tools/bloomery/hf.py` range reader fetched an 8 MiB prefix; the header ends, including
alignment padding, at byte **5,470,528**. Bytes after that boundary were ignored. Header SHA-256:
`fb0cc4b520f927f0b679193397ee0c36ca252426e7881283c5aa6de3e44a3a2b`.
The existing Python header reader and Rust `gguf::inventory_of` agree on **779 tensors** and
**25,049,218,176 encoded tensor bytes**. Shape products total **39,785,287,456 parameters** (derived).
That initial design inspection fetched only the prefix. A later local CPU oracle run used the
complete 25,054,688,704-byte GGUF (SHA-256
`63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f`).
Mainline CPU completion is reference evidence, not execution of this CUDA body.

The official repository says Apache-2.0; this exact GGUF header says `other` / `flint-preview`.
The config/header shapes and the byte-identical 2,328-byte chat templates establish those contracts,
**not equivalence of the weights or licenses**. Resolve the artifact's provenance/license with its
publisher before redistributing a fixture or describing this GGUF as an Apache-2.0 artifact. Do not
silently rewrite the header or make the engine's architecture reader a license checker.

## 2. File facts the reader must preserve

| Fact | Exact contract |
|---|---|
| Model | `general.architecture=flint`; width 2,048; 49 blocks; untied embedding/output; vocabulary 150,144 |
| Attention | 32 Q heads, 4 KV heads, head dimension 128; Q width **4,096**, not the residual width; RMS epsilon `1e-6` |
| Layer masks | Full attention and NoPE at **0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48**; the other 35 layers use SWA and RoPE |
| Window/position | 513 tokens = current token plus 512 predecessors; RoPE base 10,000; configured maximum context 262,144 |
| Dense FFN | Layers 0–1, SiLU-gated, intermediate width 6,144 |
| Routed FFN | Layers 2–48; 256 experts, top 6, each intermediate width 512; sigmoid scores; selected-weight normalization; scale 1 |
| Shared FFN | One always-active physical gate/up/down stack, intermediate width 1,024; GGUF `expert_shared_count=2` counts width units of 512, not two separately routed modules |
| QK norms/sinks | Per-head Q/K RMSNorm weights `[128]`, shared across heads, and one learned sink logit per Q head `[32]` in every layer |
| Tokenizer | `gpt2` BPE, `pre=flint`; 150,144 tokens and 149,744 merges; `add_bos_token=false`, `add_eos_token=false` |
| Control IDs | BOS 150000; PAD/EOS 150001; EOG set **150001, 150003, 150005, 150018** |

Both GGUF boolean arrays (`attention.sliding_window_pattern`, `attention.rope_pattern`) were read by
Rust and compared with the official config's `layer_types` / `no_rope_layer_indices`; they agree.
The initial two and final layers make a blind `layer % 4` rule wrong. Read and validate the arrays;
never reconstruct this model from the issue's shorthand “three sliding to one full.” The GGUF EOG
array also agrees with all four config EOS IDs, not only the scalar `eos_token_id`.

### Tensor roles

| Scope / GGUF suffix | Shape | Types actually observed in this specimen |
|---|---|---|
| `token_embd.weight`, `output.weight` | `[2048,150144]` | Q8_0, separate tensors |
| `output_norm.weight`; block `attn_norm.weight`, `ffn_norm.weight` | `[2048]` | F32 |
| Block `attn_q.weight` | `[2048,4096]` | Q8_0, Q5_K |
| Block `attn_k.weight`, `attn_v.weight` | `[2048,512]` | Q8_0, Q5_K |
| Block `attn_output.weight` | `[4096,2048]` | Q8_0, Q5_K |
| Block `attn_q_norm.weight`, `attn_k_norm.weight` | `[128]` | F32 |
| Block `attn_sinks.weight` | `[32]` | F32 |
| Dense `ffn_gate.weight`, `ffn_up.weight`; `ffn_down.weight` | `[2048,6144]`; `[6144,2048]` | Q8_0 |
| Routed `ffn_gate_inp.weight` | `[2048,256]` | BF16 |
| Routed `ffn_gate_exps.weight`, `ffn_up_exps.weight` | `[2048,512,256]` | IQ3_S, Q5_K, IQ4_XS, Q4_K |
| Routed `ffn_down_exps.weight` | `[512,2048,256]` | Q5_K, Q8_0, Q6_K, IQ4_XS |
| Shared `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`; `ffn_down_shexp.weight` | `[2048,1024]`; `[1024,2048]` | Q8_0, Q5_K, IQ4_XS |

There is **no router correction-bias tensor** in the 779 descriptors. In particular, gate/up expert
layer 2 is IQ3_S, expert-down layer 4 is Q8_0, expert-down layer 8 is Q6_K, and all three shared
projections in layer 42 are IQ4_XS. Admission must check each role and each layer; the filename's
`Q4_K_XL` is not a kernel capability. The companion evidence lists every layer/type association.

## 3. Numerical contract, not just matching dimensions

The [reference graph][ref-flint] is sequential pre-norm attention and FFN:
`a = x + Attention(RMS2048(x))`, then `y = a + FFN(RMS2048(a))`.
Q and K receive learned RMSNorm **within each 128-element head** before optional RoPE; V does not.
SWA layers use full-head NeoX pairs `(j,j+64)` for `j=0..63`; NoPE layers perform normalization
without rotation. Attention scales logits by `1/sqrt(128)`.

A learned sink is a **logit with no value vector**. For head `h`, visible scores `z_j` and sink `s_h`,
its output is `sum(exp(z_j-m)*v_j) / (exp(s_h-m)+sum(exp(z_j-m)))`, with
`m=max(s_h,max(z_j))`. Add the sink denominator **once**, not once per partial attention segment.
It consumes no KV row. A zero sink logit is not the same as an absent sink.

The [pinned routing graph][ref-route] selects the six largest sigmoid scores and weights the
selected **unbiased** scores by `p_i / max(sum_selected(p), 2^-14)`, scale 1. The real artifact has
no selection bias. An optional supported `exp_probs_b.bias` would change selection only, never the
source of the gathered weights. The shared SwiGLU output is added without a shared sigmoid gate.

Three numerical obligations must stay explicit:

1. Legacy [common router normalization][route-core] uses `sum + 1e-20`, not the reference's floor.
   This is a semantic difference for tiny scores, not a width-instance problem or a measured GPU
   failure. `Router`/`RouterRule` carries the explicit policy; the reader requests
   `SelectedSumFloor` and the common `sigmoid_floor_256_6` binding implements it. Legacy instances
   keep their exact arithmetic and `norm_arg` serves only the two legacy policies. The floor
   gate checks active tiny sums independently; GPU numerical and resource evidence is still needed.
2. Mainline [rescales Flint's activated expert input before down projection][ref-down] with
   `build_amax_src1_f16`, then restores its scale before router weighting. This protects F16/Q8
   headroom around a maximum magnitude of 1024; it is not an architectural activation clamp.
   Compare the common activation packing at that boundary and large magnitudes. Any needed finite
   range handling belongs to that common operation, with explicit arithmetic/gate evidence.
3. The shared routed owner adds `x + shared` before routed sums; mainline forms
   `(routed + shared) + x`. Accumulation order and activation formats prevent a bitwise-equivalence
   claim from algebra alone. Keep established models unchanged and gate the Flint numerical choice
   against the pinned oracle and the engine's own arithmetic referee.

## 4. Reader and admission: use the existing description

Use `ModelSpec`/`LayerSpec`, `Mixer::Gqa`, `Ffn::Dense`/`Moe`, `Shared`, plain residual and existing
[tensor roles/stems][stems]. No second schema, interpreter, capability catalog or per-token
architecture-name dispatch is needed. At NoPE layers use the existing canonical unused-RoPE
representation (`mode=None`, `dims=0`, `base=0`, no YaRN). Elsewhere use NeoX/128/10000. Preserve
both masks independently even though this artifact's masks coincide.

The reader must classify all 779 descriptors, validate exact ranks/axes and positive checked
geometry, reject unknown/misplaced tensors, and require the untied output. Bias presence becomes
`Router.bias`; it is false here. Read explicit shared width 1024; do not copy `noaux::Experts`, which
requires bias and derives width from a shared-count convention. Invalid mask types/lengths, missing
norms/sinks/shared matrices, an out-of-range block, impossible KV grouping, zero dimensions,
nonfinite epsilon/base, overflow and malformed optional bias all need named refusals.

The original reader-only landing retained `program_of=None` and `Need::Program` refusals.
The integration now has actual operation consumers and Program admission; adding a name or
loosening a type row alone would still not establish runtime support.

### Concrete common prerequisites

| ID | Original gap (design snapshot above) | Required common change / acceptance contract |
|---|---|---|
| G1 | NoPE normalization-need generation is present; the executable head128 norm-only binding is missing | Preserve the emitted normalization **or** rotation need; bind head128 normalization with `mode=None` truthfully. No identity-RoPE table. |
| G2 | Existing head128/group8 attention lacks the complete sink/SWA contract; MiMo's window pass is K192/V128 | Complete the described head128 operation in `rope_neox` / `flash_gqa`, including genuine NoPE append, sink merge, window/ring positions, prompt and slot paths. Never pad heads to 192 or copy MiMo's body. |
| G3 | Real common sigmoid256/top6 floor operation and both optional-bias metadata bindings are implemented; whole-program binding remains separate | Attach `sigmoid_floor_256_6` through `RouterEntry`, with `RouteLayer.bias=None` retaining genuine absence and no zero buffer. Price/load the BF16 router through its supported F32 widening. Run `gate-flint-router` and obtain genuine new-entry JIT pins on NVIDIA hardware. |
| G4 | Layer42 shared IQ4_XS has routed/GEMM operations but no ordinary dense site, and `BlockPlan` rejects it | Extend the **common** dense/gated block's real layout, load, launch, scratch and prompt binding. Keep the role `SharedExpert`; do not relabel it as routed or silently requantize it. |
| G5 | The common card wrapper uses Q8Act-only buffers and `LegStacks::of`; Q6_K/Q8_0 down do not both pass that path | Carry selected arms and actual down forms through `CardExperts`, `LegBufs::for_downs`, loader, planner, residency and tier. Existing Q6_K/Q8_0 arithmetic is not the missing piece. |
| G6 | Common tier accumulation supports six slots, but F32 `enqueue_handoff_tier` accepts eight | Add the six-slot F32 handoff instance in the existing owner; preserve the eight-slot entry. A DeepSeek Q8_1 handoff is not an interchangeable input contract. |
| G7 | `place_hybrid` rejects paged experts; common `Hybrid<F>` has `RESIDENT=false` | Lift matching plan/load/host-source and resident-slot plumbing in the existing adapters. Do not remove refusal before the loaded body and seat actually implement the capability. |
| G8 | Common original-template, tokenizer and four-EOG support is present through #677/#678 | Bind the existing chat/tool/stop owners when the executable seat attaches; keep the original template and GLM-format tool parser. |

Attention implementation boundary: the existing head128 ring pass and value-head128 sink merge
are promising ingredients, **not an already callable Flint operation**. Their partial segmentation
must agree: a `SEGMENTS` partition over an effective row count cannot be passed to a wrapper using
unrelated `window_segments(513)`. Bind one correct pass/merge path, not a sinkless merge followed
by another merge. A new NoPE entry is an add-class change, even if its internal helper already
accepts `ROT=0`. See [attention owners][flash] and [head normalization][rope].

### Per-role quant admission

[QROWS][qrows] and [placement kernel predicates][kernels] remain authoritative. All eight routed
role/type combinations below have host kernels and card operation entries at this pin. Eligibility
still means the selected program's gate/up **and** down arms, layout and buffers agree.

| Routed role | Observed type | Card operation / down form | Existing integration limit |
|---|---|---|---|
| Gate/up | IQ4_XS | `Iq4xs` | Common leg operation exists |
| Gate/up | Q5_K | `Q5k` | Operation exists even though the default routed-layout column is empty |
| Gate/up | Q4_K | `Q4k` | Common leg operation exists |
| Gate/up | IQ3_S | `Iq3s` | Decode **and prompt GEMM** exist; an older contrary comment is stale |
| Down | Q5_K | `Q5k` / `Q8Act` | Common leg operation exists |
| Down | IQ4_XS | `Iq4xs` / `Q8Act` | Common leg operation exists |
| Down | Q6_K | `Q6k` / `Q8Act` | G5: selected arms and loader, not absent arithmetic |
| Down | Q8_0 | `Q8_0` / `Q8Blocks32` | G5: actual buffer form and accounting |

Host rows require their declared CPU features and block widths; 2048/512 satisfy the listed block
geometries, but unsupported hardware still refuses. The current leg contract requires gate and up
types to match. If a layer's triple does not satisfy the selected intersection, keep that whole
routed layer on the supported host path and report it; residency must not promote it through a
looser rule. Qwen's wider common-leg admission can inform the lift, **not select a Qwen program**.
Attention Q8_0/Q5_K and shared Q8_0/Q5_K/IQ4_XS must be admitted by their own roles as well.

## 5. Required common-owner table

Every row from [new-model.md](../contrib/new-model.md) has a decision. **Attaches** means a design
commitment with named prerequisites, not an already implemented Flint capability. No row has an
architecture-based reason for a private replacement or omission.

| Row | Decision | Current owner | Attachment obligation |
|---|---|---|---|
| Placement plan | **attaches** | `placement::{plan_routed,plan_routed_reserving}`, `SeqTerms`/`SlotsOf`, `runtime::stores` | Actual per-layer layouts, KV/slots, scratch, residency/tier reserves and supported program rule; G5/G7. No family-owned planner. |
| Kernel table | **attaches** | `gguf::QROWS`, `placement::kernels`, `placement::accepts`, `model::arch::coverage` | Admit only callable intersections; G1–G5. Keep `Need::Program` until the complete body is bound. |
| Card expert leg | **attaches** | `gpu::{card_leg,card_experts,card_slots}`, `gpu_deepseek41::shared_routed::Routed` | `CardFamily` 256/6, width2048/expert512; existing dense lead and ungated shared composition; G3–G5. No private GLM/MiMo FFN copy. |
| Residency | **attaches** | `TierBody`/`TierShell`, `ResidencyParts`, `PlanStacks`/`ResidencyGlue` in `gpu::host` | Return real `Some(ResidencyParts)` for an enabled load, sharing slot map/source over routed blocks `2..49`; same type intersection for placement and promotion. Attach NVMe to host service **and** residency reads. |
| Expert tier | **attaches** | `gpu::host::{leg_tier,tier}`, common `CardLeg`, `shared_routed::join_rows` | G5/G6; price staging and actual down forms, test prompt and decode, include host leftovers. Preserve the existing supported-card-count refusal. |
| Prompt walk | **attaches** | `runtime::{prompt,sched}`, `gpu_deepseek41::prompt_walk::{BatchAttention,Batch}` | Bind stage/attend/FFN to common dense/routed scheduling, checked positions and ring-safe ordering; G1–G5. Existing call-end checkpoint marks do not imply arbitrary checkpoint support. |
| Resident slots | **attaches** | `gpu::model::slots::{Slots,SlotRows}`, `placement::slots`, `app::Session` | Slot-local stores/high-water/captures, model-wide weights/host tier/residency/fault word; price every requested slot and never lower `--parallel` silently; G2/G7. |
| Serve seat | **attaches** | `app::arch::Batch`, shared `serve_seats::{hybrid,tiered,rounds}` | Thin facts/registration after G7/G8; implement actual resident-slot open/step and ring-aware keep. Copying `Hybrid` with `RESIDENT=false` or MiMo's private seat is not completion. |
| Timing | **attaches** | `tools/ref/depth-serve.sh`, existing model profile/corpus and `lease_take` | The admitted HTTP seat, pinned artifacts, prediction card and common lease runner. No Flint-only timing driver or performance claim from this design. |

Current owner anchors: [planner][planner], [slots][slots], [card leg][card-leg], [card wrapper][card-experts],
[shared routed composition][shared-routed], [residency][residency], [tier][tier], [handoff][handoff],
[prompt walk][prompt], [application][app], [shared seat][seat], [timing][timing].
The historical checklist's “each family owns its expert leg” and “MiMo residency returns None”
are not current facts: the leg is common and MiMo delegates residency to its shell.

This follows [#609's composition contract](layer-composition.md): resolve concrete bindings at
load/capture time; preserve fused operations, lifetimes, streams, capture addresses and slot-local
writers. No per-token metadata interpreter or second capability catalog. PR [#666](https://github.com/midagedev/bloomery/pull/666)
was active at inspected head `6e33aee3e0826c2e130b5e72b805f54ab65e67a9`, moving Qwen plan/read/walk
seams into `chain`. Rebase implementation onto the landed ownership boundary; do not introduce a
competing abstraction. That Qwen seam is not proof of Flint or Metal support.

## 6. Placement, slots and retention

Use [common layer stores][kv-layout] and [ring rules][stores], not nominal file size as a fit test.
For per-slot context `C`, resident slots `S`, and f16 K/V, the derived KV row is
`4*(128+128)*2 = 2048` bytes. Today's 512-row append-before-read prompt contract needs
`R(C)=min(C, ceil((513+512-1)/256)*256)=min(C,1024)` rows for each sliding store.
Therefore:

`KV(C,S) = S * 2048 * (14*C + 35*min(C,1024))`.

| Per-slot context | One-slot mixed ring/plane KV bytes | One-slot all-plane KV bytes |
|---:|---:|---:|
| 4,096 | 190,840,832 | 411,041,792 |
| 8,192 | 308,281,344 | 822,083,584 |
| 262,144 | 7,589,593,088 | 26,306,674,688 |

These are **derived store terms**, checked with `runtime::stores::ring_rows`, not measured total
VRAM. Reserving only 513 sliding rows would undercount the current prompt implementation. Add
one stage RoPE table of `512*C` bytes under the current f32/128 convention, **not per slot**.
Sinks are static weights; `KvLayout.sink_row` is an unrelated sparse-indexer dummy row and does not
apply. `SeqTerms::plan_kv` already handles the live sequence's `beside` exclusion; do not charge
all shared scratch once per slot or omit actual slot-local buffers.

The encoded payload separates into **23,049,797,632 routed bytes** and **1,999,420,544 other bytes**.
The file is about **23.334 GiB**, below 24 GiB. It does not alone prove a 3090 fit or non-fit.
BF16 routers alone widen from 1,048,576 to **2,097,152 bytes per routed layer** in the existing
resident format. Add actual card packing, allocation granules, KV, RoPE, scratch, tier/residency
buffers and margins before choosing a split. Host-tier attachment is required by the integration
contract, not established merely by comparing decimal GB with binary GiB.

For layer `l`, derive each expert's encoded bytes from its actual row types:
`512*r_gate(2048) + 512*r_up(2048) + 2048*r_down(512)`, where
`r_t(k)=k/block_t*bytes_t`. Resident bytes come from `CardFormat::buffer_bytes`, not that encoded
sum. Use [common activation/leg terms][kernels]: for prompt capacity `P`, six slots/token, the leg
contains packed input for `(P,2048)`, F32 gate/up output `(6P,512)`, **all actual down forms** for
`(6P,512)`, and F32 down output `(6P,2048)`. Add the attention/QKV, router, dense6144/shared1024,
partial-merge, host/tier exchange and slot allocations according to their actual lifetimes.
The existing Q8Act-only tier helper must not underprice Q8Blocks32. Reject invalid/overflowing
geometry before calling helpers whose products are not themselves checked.

A filled 1024-row ring retains cuts of **512**, but not **513**, at context4096; the shared helper
was exercised at both sides. Retention uses `max(written_high_water,current_pos)`, and cuts do not
lower that high-water mark. Slot park/reset/swap must preserve it with store/capture identity.
The 35 unshadowed window stores are checkpoint state even though Flint has no recurrent layers.
Without integrated saved-window restoration, deeper rewind means named refusal or reset/refeed,
not invented prefix reuse. Ordinary slots do not imply partial speculative keep; preserve
`SlotRows::SETTLES_PARTIAL_KEEP=false` until its stronger restoration contract is implemented.

## 7. Tokenizer, template, tools and stopping

The actual metadata was passed to `Tokenizer::from_keys`; the exact official/GGUF template was
passed to `ChatTemplate::parse`. Current-source host probes produced:

```text
pre-tokenizer "flint" is not supported (...)
chat template: block set is not supported: set tool_preamble
chat template: |replace takes no arguments
```

The last result is an independent minimal render of `{{ 'abc' | replace('a', 'x') }}`; the official
template stops earlier at block capture. Full diagnostic strings are in the evidence JSON.
These are observed current refusals, not a claim that a reader or template fix has landed.

- **Pretokenizer:** extend the [common tokenizer][pretok] with the exact ordered split program:
  number runs of 1–3; the declared CJK/Kana ranges; then Unicode case/mark-aware words,
  contractions, punctuation and whitespace. Use the pinned [mainline regexes][ref-pretok] and
  official tokenizer as separate comparators. No normalizer, no automatic BOS/EOS and no
  `glm4`/Qwen/GPT-4o alias. Cover combining marks and range edges, not just ASCII examples.
- **Jinja:** add block `set` capture and the two-argument `replace` filter to the [existing engine][jinja].
  Macro, namespace, `.items()`, `tojson(ensure_ascii=False)` and string escapes already have common
  owners; keep the 2,328-byte original template. Validate their combination rather than install
  a rewritten Flint renderer.
- **Template behavior:** preserve NUL/think-tag removal and role-marker escaping in non-assistant
  text; tools preamble insertion after initial system messages; developer turns; reasoning
  history; raw string versus JSON nonstring arguments; and `<|observation|><tool_response>`
  results. The generation suffix is `<|assistant|>`, not an open `<think>`.
- **Tools/reasoning:** the existing [GLM key/value grammar][glmxml] matches
  `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>`.
  Bind `ToolFormat::GlmXml`, existing `ThinkTags`/`MayOpen`, and the common OpenAI argument-object
  adapter. This is source-derived reuse, not yet exercised Flint streaming/HTTP compatibility.
- **Stops:** choose the official/artifact **four-ID EOG contract**. Current [explicit-stop ingestion][vocab]
  reads `tokenizer.bloomery.eog_token_ids`, not this artifact's `tokenizer.ggml.eog_token_ids`.
  Extend common ingestion/validation for the actual key and preserve range/dedup/type invariants;
  never collapse it to scalar EOS. The pinned mainline's Flint **name-based rule** adds user and
  observation but not developer. Verify its effective loaded stop set rather than claim a
  runtime three-stop result from that branch alone. Record any discrepancy separately from
  teacher-forced logit comparison; do not silently discard developer/150018 to match a heuristic.

## 8. Ordered implementation hand-off and acceptance

This design PR changes no runtime registration, admission row, gate threshold or numerical rule.
The next contributor should implement the following dependency order, keeping each common change
with its real caller and independent evidence:

1. **Reader and host contract.** Add the Flint reader/architecture row, exact roles, both masks,
   explicit widths and optional-bias handling. Correct G1 and the common tokenizer/Jinja/EOG
   gaps. Reader-only support remains `Need::Program`; no advertised runnable seat.
2. **Reference and fixture identity.** Pin mainline build/model/header identities. Use the existing
   fixture machinery with `PerLayer` for both arrays. Preserve both dense lead layers, full/SWA
   dense/MoE combinations and transitions, and every observed `(role,type)` pair—not a convenient
   homogeneous subset. In particular retain layer2 IQ3_S, layer4 Q8_0 down, layer8 Q6_K down and
   layer42 shared IQ4_XS. Resolve artifact/redistribution provenance before publishing fixtures.
3. **Common numerical/operation prerequisites.** Complete G2–G5: head128 norm/NoPE/sinks/SWA;
   unbiased sigmoid256/top6 with explicit normalization policy; shared IQ4_XS; actual card-leg
   arms/down forms and matching checked placement/scratch. No admission before callable bindings.
4. **State and tier integration.** Attach common prompt, host service, residency, NVMe and slots;
   implement G6/G7 together with their accounting and refusals. Preserve capture/stream lifetimes,
   ring high-water retention and the host leftovers of a two-card split.
5. **Body, application and seat.** Bind the existing scheduler/composition and thin family facts;
   enable the program/seat only when coverage and load agree. Complete end-to-end gates below,
   then the existing profile-driven timing runner. Do not call completion after a compiling reader.

### Required evidence before runtime support

| Boundary | Consumer-visible acceptance |
|---|---|
| Reader / coverage | Missing/malformed masks and required tensors, axes/counts, optional-bias errors, unknown names, zero/nonfinite/overflow geometry refuse by name; 49 norm needs, 35 rotary needs, 49 sink needs. Unsupported program/site remains a refusal. |
| Tokenizer / chat | Byte-exact mainline and official-tokenizer comparisons on Unicode/number/tag corpus; original-template comparison against Jinja2 for roles/tools/reasoning/escaping; streamed versus whole GLM calls and malformed/truncated errors; all four EOG IDs, no added BOS/EOS. |
| Attention | Head-local versus flattened RMS discrimination; half-pair RoPE and genuine NoPE; sinks once per reduction, zero versus absent sink; lengths512/513/514 and wrap1024; plane/ring, prompt/decode, eager/replay and multiple-slot isolation. |
| Router / FFN | Top6 IDs/weights, ties, absent bias, optional selection-only bias, sums around `2^-14`, large activated values around reference rescaling; dense/shared/routed output and explicit residual-order comparison. No threshold loosening. |
| Quant / placement | Every observed role/type and mixed triple; gate/up mismatch, Q6_K and Q8_0 down forms, shared IQ4_XS; plan-accepted equals loader-accepted, budget-floor/overflow refusals and host/card/tier conservation. |
| Stores / slots | Context boundaries1/512/513/1023/1024/1025 and slots1/2/3; stage RoPE charged once; requested slot count retained; filled-ring cuts512/513, repeated cuts, divergence, park/reset/swap and high-water preservation. |
| Host / residency / tier | Mixed host/card picks; promotion/demotion constrained by the same capabilities; NVMe fill/reuse/eviction and residency-source lifetime; sidecar refusals; two-card **prompt and decode** with six-slot F32 handoff and stable merge. |
| End to end | Pinned-mainline teacher forcing at dense, first routed, sliding/full transition and final layers; exact real mixed-quant artifact, recorded backend/KV/flash settings; HTTP/direct outputs and checkpoint-resume behavior. A header census is not this gate. |

Pure lifts need the existing canonical/no-JIT move proof plus eager/capture/prompt/slot behavior.
New router/NoPE/handoff entries are **add-class** changes with base-plus-intended-entry scans,
spill/resource evidence and FAIL-first owning gates. Numerical guard/order changes are numerical
changes, not moves. Run the actual `just affected BASE` selection and owning existing shared
routed, Qwen/MiMo attention, residency and tier gates; none replaces a Flint oracle set. GPU and
lease-backed timing evidence require a suitable machine and have **not** been produced here.

Future bookkeeping belongs to the existing owners: `model::arch::registry` and its generated
`tools/arch-registry.tsv`; `model-hf`'s fixture CLI/`FIXTURE_SPECS`; `tools/fixture-dirs.tsv`;
`refset::arch` and its pinned oracle/dump/profile; and `tools/{families,gate-class,owners,gate-paths,gate-batch-resources}.tsv`.
Update PTX/unsafe/resource records only from the later implementation's actual proof, not this design.

## 9. Evidence produced for this design

Executed locally, host-only:

- Bounded HTTP-range read at the pinned GGUF revision through the existing Hub reader and parser;
  actual header hash, full tensor descriptors/types/shapes and encoded-byte accounting.
- `gguf::inventory_of` on that same prefix; independent array extraction and identical tensor-byte
  total. Config comparisons for both masks, EOG IDs and vocabulary; exact official/GGUF template bytes.
- The three observed tokenizer/template/filter refusals in §7.
- `runtime::stores::ring_rows` at the nine contexts in the JSON, and `cut_depth_keeps` on both sides
  of the filled-ring512/513 boundary. The displayed KV formula uses those actual returned row counts.

No permanent test of documentation wording, fabricated model output or sparse zero-weight inference
was added. Throwaway host probes used the public APIs above and were not installed as model support.
The companion JSON is a source/evidence snapshot, **not** a second runtime capability registry.

To reacquire and check the measured header without downloading the full weights, run from repository
root with a scratch directory; this reuses the tree's one Python parser and the pinned JSON identity:

```sh
python3 -B - <<'PY'
import hashlib, importlib.util, json, pathlib, tempfile
root = pathlib.Path.cwd()
e = json.loads((root / 'docs/research/flint-attach-evidence.json').read_text())
spec = importlib.util.spec_from_file_location('hf', root / 'tools/bloomery/hf.py')
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
g = e['gguf']
with tempfile.TemporaryDirectory() as tmp:
    hub, prefix = hf.Hub(tmp), str(pathlib.Path(tmp) / 'header.bin')
    meta, tensors, _ = hf.read_header(
        lambda a, b: hub.range(g['repo'], g['file'], a, b, g['revision']),
        prefix, g['file_bytes_from_hub'], 64 * hf.MIB)
    assert meta is not None
    end = hf.gguf_ranges().header(prefix, size=g['file_bytes_from_hub'])[2]
    assert end == g['header_bytes']
    assert hashlib.sha256(pathlib.Path(prefix).read_bytes()[:end]).hexdigest() == g['header_sha256_measured']
    assert len(tensors) == e['arrays_and_counts']['tensor_count']
    total = sum(hf.gguf_ranges().nbytes(g['file'], n, t, d) for n, t, d, _ in tensors)
    assert total == e['tensor_payload_bytes_derived']
    print(f"header={end} tensors={len(tensors)} encoded_bytes={total}: verified")
PY
```

The Python reader intentionally skips array values. Their recorded values came from
`gguf::inventory_of(...).meta`, not `None` values treated as absent GGUF facts. The recipe above
revalidates the **whole header hash**, including those arrays, without a second parser.

## 10. Single-card NVIDIA acceptance and remaining numerical boundary

The runtime at `313b9f0f88c90fcfae2365a4af53cd077762eb7c` was exercised on one RTX 3090,
cc8.6, driver 595.91.07 (CUDA API 13.2), with nightly-2026-08-28, cuda-oxide `87bc9518`,
LLVM 21.1.8 and CUDA toolkit 13.3.0 / ptxas 13.3.33. The complete 25,054,688,704-byte
GGUF was downloaded separately on that runner and its full SHA-256 matched
`63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f`.

- `just gate-gpu-flint-attn` and `just gate-gpu-flint-router`: exit 0, including loaded
  NoPE/NeoX ring consumers, floor/optional-bias routing, capture/replay and named faults.
- `just gate-gpu-flint-lib`: exit 0; the full-file capture/slots/reset/ring-boundary test
  ran with **1 passed, 0 failed, 0 ignored**. Its bitwise consistency is not a mainline oracle.
- The real HTTP seat loaded 49 layers and two resident slots, exposed the original template,
  generated English and Korean turns to EOG 150003, split reasoning, parsed a required
  `get_weather` call with `{"location":"Seoul"}`, and preserved messages under concurrent
  requests. The four stop spellings tokenized to the four expected IDs; natural emission of
  all four stop IDs was not tested.
- A full `generate_ds41` scan and the pre-pin spill gate JITed all 436 entries without a
  resource cache. The only ten violations were the new unpinned entries, each measured
  `spill=0, jit_local=0`; the 26 vision entries passed. These observations supply only the
  ten `sm_86`/cc8.6 rows. The three cc12.1 tables still need thirty genuine measurements.

### Independent greedy comparison remains partly red

The English raw continuation exactly matched pinned mainline CPU
`a37cc46104eb50900d5ab4a9ba02a0abdac039db`. The Korean continuation did not: the first
different generated ID is index 74, after the same 22 prompt IDs and 74 generated IDs
(next absolute position 96). Mainline chooses 26007 (`라`); Bloomery chooses 29263 (` 정`).
Both eventually answer `4`, which does not turn the comparison green.

A standalone consumer linked to the preserved mainline library replayed that original
prompt-then-single-token trajectory with zero preceding raw-argmax mismatches. Its complete
150,144-logit rows at context 1024 and 2048 were bit-identical. At the divergence:

| Observation | Margin `logit(26007) - logit(29263)` | Selected ID |
|---|---:|---:|
| Mainline CPU, original trajectory | +0.608115554 | 26007 |
| Bloomery probability diagnostic, original trajectory | -0.018128157 | 29263 |
| Bloomery, same 96 IDs supplied as one prefill | +0.361473739 | 26007 |

The GPU margins are differences of reported pre-sampling log probabilities; the diagnostic
reproduced the original first 75 generated IDs. Probability collection reads each row and
uses plain per-token stepping, rather than establishing equality of every execution mode.

This is not an exact tie or a demonstrated context-capacity defect. The common FFN owner
explicitly uses F32-activation Q8_0 GEMV for narrow/decode rows and quantized-activation
GEMM for wide prompt units (`shared_routed.rs`, `GATED_FROM=9`). Mainline also quantizes
the output head's activations and BF16 router operands differently. These are source-backed
numerical differences, not a proven localization of the Korean flip. No precision mode,
compiler pin, tie rule, tolerance or acceptance criterion was changed to force matching text.
Comparable follow-up traces must retain the original prompt/decode trajectory and can use
the existing 49-layer `Body::taps` and mainline `l_out` callbacks.

Keep the integration Draft. Korean numerical/oracle acceptance, cc12.1 resource rows,
affected common-owner coverage beyond the exercised consumers, two-card operation, dynamic
residency/NVMe, streamed HTTP and spontaneous tool selection are not established by this run.

## 11. Current-main integration and trajectory instruments

The first linear integration candidate, `4dadee0195fb652608be59ffecc033a4643ad1eb`,
starts at main `43073091be21e48cda40d79fe6dae1dfcfd643d5`. The follow-up replay
starts at main `af74ea46b7f3e47692c2afae09eb64fdddf71461`. At that stage neither replay
replaced the published #799 head; section 12 describes the later history-preserving cutover.
Fixed-source comparisons keep the original production and candidate images separate, and
receipts name each image. Section 10 is not acceptance of either replay or the later cutover.
The placement integration keeps both the cached upload `Footprint` and candidate-selected
tier scratch/reserves. No compiler pin, numerical band or model admission was widened.

### The common union must retain selected formats and recount its allocation

`model::placement::union_plan` now owns the shared one-stage/one-tier reference transform.
It retains the actual selected card format, refuses incompatible formats, preserves host
assignments and KV, and recomputes expert bytes, upload-heap rounding, selected reserves and
actual-stage headroom. The artificial budget that split the experts is not the reference's
physical capacity. The old shared helper is removed and its consumers use the common owner.
The standalone DS41 and Qwen3.8 transforms have different contracts and remain separate.

A public host-only consumer over the complete pinned GGUF reproduced the former generic
transform's `NoCardFormat` refusal on `blk.2.ffn_gate_exps.weight` (`IQ3_S`). The corrected
transform preserves the program-selected format rather than admitting IQ3_S generically.
With the workstation's 3090 stage/A6000 tier figures, 1024 context and an 8 GiB per-card
planning budget, its union contains 5,197 experts and 9,959,514,112 expert bytes, equal to
the assigned rows, with no plan violations. This is real-file planning, not device loading.
The pure regression also covers mixed selected down forms, heap/reserve recounting,
unchanged generic refusal, conflicting formats and insufficient union capacity.

The common mixed card/tier hardware consumer stages its fixture through `Weights::load_rows`
using `accepts::routed_rule(FAMILY.program)` and the shared segment buffer arithmetic.
Neither `Weights::load_where` nor the generic `routed_row` admits selected-only IQ3_S:
assigning a routed role alone is insufficient. The fixture keeps all mixed-format
step/pass/batch/tier bit comparisons and byte conservation, without widening generic
admission. The obsolete Solar2 Q6_K-down rejection is removed: the current selected owner
already admits that arm. Mismatched gate/up rejection remains exercised.

### CPU and GPU use one forced-ID diagnostic artifact layout

`tools/ref/flint_trace.cpp` and the `bloomery-gpu-flint` example `trace_flint` consume exactly
96 whitespace-separated decimal IDs. The original HTTP `Slot::close_call` in
`crates/serve/src/genloop.rs` prefills the first 21 prompt IDs, then calls `next` for the
22nd ID before the 74 forced continuation IDs. Both consumers reproduce that boundary.
They do not retokenize, sample, or stop at an input EOG. The 75 observations save processed
positions 21 through 95, predicting positions 22 through 96; the prefix itself is not an
observation. `--prompt 22` names the complete prompt, not a 22-ID batched submission.

The format is `flint-trajectory-v2`, not a reference-set family or accuracy gate.
The former v1 consumer submitted all 22 prompt IDs as one batch: its controls remain
historical evidence but do not reproduce the actual HTTP prompt/step boundary. Do not
relabel v1 arrays or compare them as if the submission contracts matched.

- `tokens.ids` and `steps.tsv`: exact inputs and each call's processed/predicted position
  and raw argmax.
- `logits.f32`: little-endian F32 `[75][150144]` in every arm.
- With taps: `layers.f32` `[75][49][2048]` and `norm.f32` `[75][2048]`.
  Untapped arms explicitly mark those tensors absent.
- CPU `tensors.tsv` records actual node types, dimensions, strides and selected rows.
  GPU artifacts retain the actual expert assignments, loaded slots, device/plan facts,
  graph nodes and completion counters.
- A new output directory is required. `MANIFEST.tsv` is published only after successful
  capture; a partial run cannot masquerade as complete. External receipts identify the
  model, source/diagnostic overlay, toolchain, libraries and binary hashes.

Build the CPU consumer against the preserved mainline revision
`a37cc46104eb50900d5ab4a9ba02a0abdac039db`, with shell variables naming its source,
built library directory and the new diagnostic executable:

```sh
c++ -std=c++17 -O2 -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  tools/ref/flint_trace.cpp -L"$LLAMA_LIB" -Wl,-rpath,"$LLAMA_LIB" \
  -lllama -lggml -lggml-cpu -lggml-base -o "$CPU_TRACE"
CUDA_VISIBLE_DEVICES= "$CPU_TRACE" -m "$MODEL" --ids "$IDS" \
  --prompt 22 --ctx 1024 --out "$CPU_OUT"
```

The preserved **v1** CPU callback arm was actually run on the complete model: 75 calls, 3,675 layer vectors,
zero preceding argmax mismatches, and final argmax 26007. Its complete final-logit SHA-256
is `083c8aa95aab025c0183c2ee3c4788642499533d5a2ee4e280c1125302c23bf9`, bit-identical
to the preserved callback-free original row. The model was rehashed in this round.
The separate untapped control also ran: all 75 complete 150,144-logit rows are
bit-identical to the callback arm, whole-file SHA-256
`7f2e476bd7fbf33faf6548eedadbb89b1844c50e0a55d7fc8aa25b297fbaff4a`.
These are results of the first candidate's CPU consumer, not new GPU results.

The corrected **v2** CPU callback and callback-free consumers also ran on the complete
model. Their 75 full-logit rows are bit-identical, SHA-256
`4ef35ee7390341e640a6d38c39a010643f51a853fe743b29a5da989c7b8357b5`.
At next position96 the argmax is26007 and its margin over29263 is+0.238318443.
V1/v2 CPU arrays are not bit-identical; call16 has a different argmax. This is evidence
that the submission boundary matters, not a GPU fix or permission to waive the oracle.

The GPU build uses the pinned oxide toolchain. Actual capture runs belong inside an
authorized runner configuration with its normal card locks and bounds; `TRACE` names
the compiled example in that checkout's target directory:

```sh
cargo oxide build --arch sm_86 -- -p bloomery-gpu-flint --release --example trace_flint
bash tools/gpu-gate.sh --executable "$TRACE" -m "$MODEL" --ids "$IDS" \
  --prompt 22 --ctx 1024 --slots 2 --mode graph --out "$GPU_OUT"
```

Use separate new output directories for all four mode/instrumentation combinations:
Eager/untapped, Eager/tapped, Graph/untapped and Graph/tapped (`--taps`).
Check full-logit neutrality within each mode as well as Eager/Graph agreement.
Arming taps forces Eager; the consumer explicitly restores the requested mode afterwards,
and Graph decode must actually capture. The instrument admits only the existing
card-budget lever; do not assume other inherited levers were applied.
Context 1024 is per slot, two slots total 2048, matching the original HTTP setup.
Compare the actual per-layer placement, not just model identity or aggregate counts.
The diagnostic's all-false `HostCfg` matches the original `r8=off` and `HOST_LOCK=0`
settings, but page population/discard differ from the original HTTP defaults. Record that
difference; it is neither an established numerical cause nor permission to skip direct-B
boundary reproduction. `norm.f32` is the weighted RMS output before head activation
quantization, not an RMS-only row or quantized activation bytes.

Diagnose the original `313b9f0` production source with the separately recorded example
overlay first; then bridge the same controls onto the integration candidate. A new binary
on a larger card is not automatically the original 3090 trajectory. A control mismatch
stops causal attribution. If layer outputs agree but the final norm/head does not, isolate
that operation on matching inputs; intended quantization is neither a demonstrated bug
nor automatic permission to waive the Korean flip.

### Dedicated real two-card consumer

`gate-gpu-flint-twocard` runs
`two_card_tests::hw_two_card_full_file_union_capture_publication_fault_reset`.
It requires the complete pinned GGUF in inner `BLOOMERY_MODEL`, two distinct visible
devices, and actual expert assignments to both cards at every routed layer. It compares
batched prompt and changed-input decode logits/layer rows with the same plan's one-card
union, checks ranked six-slot publication and stage/tier capture/completion, then raises
a real quantizer fault on the tier GPU while the stage is clean and checks propagation,
poison, refusal, reset and refeed. No unsafe fault-injection API is added.

The one-card library recipe keeps its exact original test filter. Both full-file recipes
refuse the fixture tier. The two-card recipe takes the existing `both`/big-load runner
path and has a derived host-resource ceiling, not a measured RSS claim. It does not prove
dynamic residency/NVMe, loss/deadline recovery or a Korean oracle result.

Require fresh NVIDIA evidence tied to each candidate's actual source, model, placement
and executed hardware case; compiled/ignored tests are not such evidence. Preserve
existing workloads unless their owner authorizes a separate window. Physical cc12.1
resource values must be measured, never inferred from an sm_86 build or cc8.6 run.

## 12. Shared-contract integration and same-input head probe — 2026-10-26

The integrated contracts started from main
`15778e2cb2dd7af428d2dd14259e6b0ea8ca6b89` and were reconciled with pinned main
`cb8d92ebddd1ce1a325a15450c06a01f57ecf88c`'s projection and stream interfaces.
They retain Rust1.99.1 and cuda-oxide9df9e7ce / repository backend patch31ab6a1d3f58,
not a model-specific compiler override. Older nightly/87bc arrays remain evidence
of their own source and toolchain only.

Current-main's prequantized leg input and metadata-only `Common` remain separate.
Both ordinary and prequantized entry points retain the loaded `AmaxF16` protection state.
The already-activated card leg normalizes with `rule=None`: its second device ABI operand
is unread, so it borrows the distinct existing down buffer rather than allocating or copying
a fake input. A hardware regression compares ordinary input with prequantized captured replay,
including changed amplitudes, Direct/AmaxF16, full/partial batches and skipped/tail canaries.
Historical integration source `082cf7ffac3bb5b5d574b489062732735985feeb`
passed native all-target Clippy and formatting, with three warning lines against ratchet three.
Its host library run passed chain23, GGUF67, placement206 and model563 (five model tests
ignored), with zero filtered tests. Common-name and unsafe ownership checks also passed.
On RTX3090 cc8.6 / driver595.91.07, its protected-prequantized, mixed/empty-tier and
complete-GGUF consumers each passed one hardware case, zero failed/ignored. Its attention
and router executables also exited0. These are source-specific results, not execution
of later ownership/API repairs or whole-model oracle acceptance.

The current-main code reconciliation at `49b11b36d4a1de2e140010c5a2d2c9afc3e2671a`
passed native formatting/all-target Clippy (zero warning lines against ratchet three),
model565 and placement206, with five model cases ignored and zero filtered.
The common-name check found1337 lines against the existing1400-line allowance;
all39 unsafe ownership rows matched their pins. These are host checks, not NVIDIA execution.
The merge retains IQ4_XS's selected `_sel` dispatch on the supplied stream and its
q8_1 launch accounting. The Qwen3.8 admission fixture accepts its projected sites while
retaining the K=640 shared-down refusal. IQ4 activation bands were not relaxed,
and the removed source-text wiring guard was not restored.

Every recipe that builds the shared `bloomery-serve` binary carries its required-feature
union, including `flint`; this also applies to newly merged consumers such as
`gate-gpu-llama-serve`. Recipe validation and actual model-fence opening must agree
before a batch can schedule that consumer. This configuration check is not a GPU serve run.

### Measured head/upstream separation, not a precision fix

A throwaway probe used the preserved mainline CPU backend
`a37cc46104eb50900d5ab4a9ba02a0abdac039db`, four threads, and the original complete
Q8_0 output matrix (2048 × 150144). It evaluated both CPU and GPU final-normalization
vectors from the matched-boundary v2 trajectory. Its CPU-input controls reproduce the
original complete 150144-logit rows bit for bit at both calls; this validates the cross-input
comparison rather than a dequantized/reduced-matrix approximation.

Here `HC`/`HG` are the CPU/GPU head implementations and `xC`/`xG` their saved norm inputs.
Each cell is the first named token's logit minus the second's:

| Predicted position / token pair | HC(xC) | HC(xG) | HG(xG) | HG(xG) − HC(xG) | HC(xG) − HC(xC) |
|---|---:|---:|---:|---:|---:|
| 38 / 86671 − 55591 | +0.164879680 | −0.166242123 | −0.142640829 | +0.023601294 | −0.331121802 |
| 96 / 26007 − 29263 | +0.238318443 | +0.005065918 | −0.018128157 | −0.023194075 | −0.233252525 |

At position96, the CPU head on the GPU norm selects26007, whereas the actual GPU head
selects29263. At position38 the CPU head on the GPU norm still selects55591. A head-only
change therefore cannot resolve the whole trajectory. The columns separate head-implementation
and upstream contributions; they do not identify the first upstream operation or attribute all
head difference to one quantization step. No production precision, tolerance, tie rule or oracle
criterion was changed, and no existing GPU arm was replayed for this CPU-only probe.

These source-specific cross-input results are not a run of a later integration.
Whole-model acceptance still requires exact-source runtime proof, disposition of
the Korean numerical/oracle mismatch, genuine physical cc12.1 resource measurements
and remaining affected common/residency/tier/NVMe coverage. Component proof is not
whole-model admission, ROCm support or a performance result.


[needs]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/models/src/need.rs#L505-L531
[stems]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/model/src/arch/stems.rs#L61-L125
[coverage]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/model/src/arch/coverage.rs#L1654-L1770
[route-core]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/route_core.rs#L43-L59
[flash]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/flash_gqa.rs#L5575-L5664
[rope]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/rope_neox.rs#L2059-L2127
[qrows]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gguf/src/qtype.rs#L84-L118
[kernels]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/placement/src/kernels.rs#L96-L280
[planner]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/placement/src/placement.rs#L2237-L2288
[slots]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/placement/src/slots.rs#L39-L147
[card-leg]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/card_leg.rs#L180-L248
[card-experts]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/card_experts.rs#L275-L330
[shared-routed]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu-deepseek41/src/shared_routed.rs#L475-L644
[residency]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/host/served.rs#L28-L61
[tier]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/host/leg_tier.rs#L342-L473
[handoff]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu/src/host/handoff.rs#L1181-L1235
[prompt]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu-deepseek41/src/prompt_walk.rs#L201-L254
[app]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/app/src/arch/mod.rs#L95-L218
[seat]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/gpu-gates/src/bin/shared/serve_seats/hybrid.rs#L41-L99
[timing]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/tools/ref/depth-serve.sh#L465-L482
[kv-layout]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/model/src/arch/place_hybrid.rs#L403-L592
[stores]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/runtime/src/stores.rs#L47-L112
[pretok]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/tokenizer/src/pretok.rs#L480-L562
[jinja]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/jinja/src/lib.rs#L480-L548
[glmxml]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/serve/src/glmxml.rs#L378-L445
[vocab]: https://github.com/midagedev/bloomery/blob/e53cabebbb9b16249c81b7c26a026da1670f7d9d/crates/tokenizer/src/vocab.rs#L330-L351
[ref-flint]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/models/flint.cpp#L80-L194
[ref-route]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/llama-graph.cpp#L2214-L2345
[ref-down]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/llama-graph.cpp#L2500-L2544
[ref-pretok]: https://github.com/ggml-org/llama.cpp/blob/a37cc46104eb50900d5ab4a9ba02a0abdac039db/src/llama-vocab.cpp#L284-L291
