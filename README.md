# CSI Feedback Autoencoder (CsiNet-style)

A PyTorch reproduction of **CsiNet** (Wen, Shi, et al., *"Deep Learning for Massive MIMO CSI
Feedback,"* IEEE Wireless Communications Letters, 2018) for compressing wireless channel state
information (CSI), trained and evaluated on a synthetic clustered multipath channel model.

## 1. The problem, from first principles

In a massive-MIMO base station, the transmitter needs to know the **downlink channel** (how the
signal it sends will be distorted on its way to the user) so it can precode transmissions
correctly. There are two ways it can find this out:

- **TDD (Time-Division Duplex):** uplink and downlink share the same frequency, just at different
  times. Within one *coherence time* (the window over which the channel doesn't change much), the
  uplink and downlink channels are essentially the same physical channel (channel reciprocity).
  So the base station just measures the signal the user sends it, and infers the downlink channel
  for free.
- **FDD (Frequency-Division Duplex):** uplink and downlink use *different* frequencies. Different
  frequencies propagate differently (different fading, different multipath interference), so
  reciprocity doesn't hold. The base station has no way to infer the downlink channel from the
  uplink. Instead, **the user equipment (UE) has to measure the downlink channel itself and report
  it back to the base station** — this report is the "CSI feedback," and most commercial cellular
  systems today are FDD, so this is a real, universal cost.

The channel is represented as a matrix `H` (subcarriers × antennas, for an OFDM/MIMO system) —
reporting it naively means sending the whole matrix back over the (scarce) uplink, which is
expensive. **CsiNet's idea:** train a neural-network autoencoder where the encoder runs on the UE
(compressing `H` into a short codeword) and the decoder runs at the base station (reconstructing
`H` from the codeword). Only the codeword needs to be fed back.

### Why is a channel matrix compressible at all?

A real wireless channel is not random noise — it's the sum of a handful of physical propagation
paths (reflections off a few walls/buildings, a handful of scattering clusters), each with its own
angle and delay. If you transform the channel matrix into the **angular-delay domain** (via a 2D
DFT — one transform over antennas → angle-of-departure, one over subcarriers → delay), nearly all
of the energy concentrates into a small number of entries corresponding to those few real paths.
That sparsity is *why* compression works, and it's why this project generates synthetic training
data from a **clustered geometric multipath model** (Saleh-Valenzuela-style: a few random
scattering clusters, each with several closely-spaced sub-paths) rather than i.i.d. Rayleigh
fading, which has no such structure and would defeat the entire premise.

## 2. Pipeline, end to end

```
clustered multipath generator  →  2D DFT to angular-delay domain  →  truncate to 32 delay taps
        (channel_model.py)              (transform.py)                    (transform.py)
                                                                                  │
                                                                                  ▼
                                                                    CsiNet encoder (UE side)
                                                                    conv → flatten → dense(M)
                                                                                  │
                                                                          codeword (length M)
                                                                                  │
                                                                                  ▼
                                                                   CsiNet decoder (BS side)
                                                             dense → 2× RefineNet residual blocks
                                                                                  │
                                                                                  ▼
                                                                    reconstructed channel matrix
```

**Compression ratio** `CR = M / 2048` (2048 = 2 × 32 × 32, the real+imaginary truncated matrix
size), swept over `{1/4, 1/16, 1/32, 1/64}`.

**Metric:** NMSE (Normalized MSE), reported in dB — more negative is better:

```
NMSE_dB = 10 · log10( mean_i[ ‖H_true,i − H_pred,i‖² / ‖H_true,i‖² ] )
```

A trivial "predict all-zero channel" baseline scores exactly **0 dB** by this definition — it's
the sanity-check floor everything else should beat.

### Two synthetic scenarios

| | clusters | rays/cluster | angular spread | delay spread | energy captured in 32 taps |
|---|---|---|---|---|---|
| **indoor** | 8 | 10 | wide (10°) | long tail | 0.894 |
| **outdoor** | 3 | 6 | narrow (3°) | short tail | 0.958 |

Indoor has richer scattering (more clusters, wider angular/delay spread), so more energy leaks
past the 32-tap truncation window even before any neural compression happens — it's the harder
scenario by construction, matching the qualitative claim in the original paper.

### Architecture (per-layer)

**Encoder:** `Conv2d(2→8,k3)→BN→LeakyReLU → Conv2d(8→16,k3)→BN→LeakyReLU → Conv2d(16→2,k3)→BN→LeakyReLU`
→ flatten (2048) → `Linear(2048→M)` (raw codeword, no activation). Deeper than the original
paper's single `Conv2d(2→2)` layer — see §4 for why.

**Decoder:** `Linear(M→2048)` → reshape `(2,32,32)` → 2× **RefineNet unit** → final
`Conv2d(2→2,k3) → BatchNorm` (see §3 for why there's no final Sigmoid, unlike the original paper).

**RefineNet unit** (residual refinement block):
`Conv2d(2→8)→BN→LeakyReLU → Conv2d(8→16)→BN→LeakyReLU → Conv2d(16→2)→BN`, then
`LeakyReLU(conv_output + identity)`.

## 3. Issues found during development (and how they were fixed)

Three real bugs were caught by deliberately gating each pipeline stage against a physical sanity
check, rather than trusting that "the code runs" meant "the code is right." All were invisible to
the unit tests (which used random, not physically-structured, data) and only surfaced once real
generated channels were run through the full pipeline.

### Bug 1 — DFT sign convention (delay-domain energy did not concentrate)

The angular-delay transform used a *forward* DFT (`exp(-j·2π·d·k/n)`) where the physically correct
operation is an *inverse* DFT (`exp(+j·2π·d·k/n)`), because the channel was generated in the
frequency domain as `H(k) = Σ paths · exp(-j·2π·k·τ/N)` — i.e. `H(k)` is already the forward-DFT
of a sparse delay-domain impulse train, so recovering that sparse structure requires the *inverse*
transform. Applying another forward transform on top scrambled the delay axis instead of
concentrating energy near the true (short) path delays.

**Symptom:** the printed diagnostic (`energy capture ratio at trunc=32`) was ~0.06 instead of the
expected ~0.85–0.95 — almost all energy was landing outside the truncation window, for *both*
scenarios, independent of their actual delay spread.

**Fix:** flip the sign in the delay-axis transform matrix (`transform.py`). After the fix, indoor
→ 0.894 and outdoor → 0.958, and the physical ordering (indoor < outdoor) — which coincidentally
still held even with the bug, since both scenarios were equally broken — became a real, correctly
-earned result rather than a coincidence.

### Bug 2 — complex-arithmetic shift bug (imaginary channel silently miscalibrated)

`to_network_input` computed `x = (h_complex/scale + 1) / 2` intending to shift *both* the real and
imaginary parts into `[0, 1]`. But adding a real scalar (`1`) to a complex NumPy array only shifts
the *real* part — the imaginary part is untouched by the `+1`, then still divided by `2`. So the
real channel correctly landed in `[0, 1]`, while the imaginary channel silently ended up in
`[-0.5, 0.5]`.

**Symptom:** caught immediately by a unit test asserting `x.min() >= 0` — a good example of why an
explicit range check is worth writing even when a transform "looks obviously right" by eye.

**Fix:** compute the shift explicitly on `.real` and `.imag` separately rather than relying on
NumPy's complex broadcasting.

### Bug 3 — normalization scheme made the target un-learnable (the big one)

This one didn't fail loudly — it failed by training "successfully" (loss went down every epoch)
while never beating the trivial 0 dB baseline, even with **zero compression** (`CR=1`, i.e. no
bottleneck at all). That contradiction — a fully-capable autoencoder that can't even learn the
identity function — was the signal that something upstream was wrong, not the model or training
loop.

**Root cause:** the per-sample input scale was defined as `s = max(|h|)` (the single peak
magnitude in the 32×32 matrix), intended to map every sample into `[0,1]` for a `Sigmoid`-ended
decoder (mirroring the original paper's design). But our clustered channel is *sparse* — a
peak-to-RMS magnitude ratio of ~6.7× was measured empirically — so dividing by the peak crushed
the "typical" entry down to a tiny sliver near the center of `[0,1]` (measured input std ≈ 0.055).
Per-element MSE loss then had almost no gradient signal for the handful of large, *informative*
entries, since they were vastly outnumbered by near-constant background entries the network could
already match by doing nothing.

**Diagnosis process** (this is the part worth remembering for next time): rather than guessing,
each hypothesis was tested in isolation —
1. Removed the compression bottleneck entirely (`CR=1`) — still failed to beat 0 dB, ruling out
   "compression is just hard" and pointing at something more fundamental.
2. Directly compared the trained model's normalized-domain MSE against the trivial
   "always predict the mean" baseline's MSE on the same data — the model was *worse* than the
   trivial baseline, confirming this wasn't a metric/inversion bug, but a real optimization
   failure.
3. Measured the actual peak/RMS ratio and per-radius energy distribution of the generated
   channels, which explained *why*: most of the input signal's dynamic range was being wasted.
4. Overfitting a tiny 50-sample set for 500 epochs still plateaued at a bad NMSE — until the
   learning rate was raised 10×, at which point it converged cleanly. This isolated a *second*,
   compounding issue: the default learning rate (1e-3, copied from the paper's setup for very
   different, non-sparse real-world data) was simply too slow for this loss landscape.

**Fix:**
- Switched the per-sample scale from **peak magnitude** to **RMS magnitude** (`transform.py`) —
  typical entries now sit near unit scale instead of collapsing toward the center.
- Removed the decoder's final `Sigmoid` (`models/csinet.py`) — RMS-normalized values aren't
  naturally bounded to `[0,1]`, and a `Sigmoid` would just saturate/clip the large, informative
  entries we actually care about reconstructing.
- Raised the default learning rate from `1e-3` to `1e-2` (`train.py`).

After all three fixes, the same `CR=1` sanity check went from **+1.1 dB** (worse than trivial) to
clean, well-behaved training — and every real compression ratio in the final sweep clears 0 dB by
a comfortable margin.

A fourth issue, of a different character — not a bug, but an under-powered architecture — showed
up once results were compared against the PCA baseline (single conv layer beaten by plain linear
PCA at CR=1/4); see the "Neural network vs. PCA baseline" discussion in §4 for that one.

### Bug 4 — a glob pattern silently let quantization results overwrite the real CR-sweep table

`scripts/make_plots.py` loaded per-CR results with `glob(f"{scenario}_cr*.json")`, filtering out
only paths ending in `_quant.json`. Once `scripts/qat_finetune.py` started writing
`{scenario}_cr{cr}_b{bits}_qat.json` files into the same directory, those matched the glob too —
they weren't `_quant.json`-suffixed, so the filter let them through — and because they share the
same `{"cr": ..., "test_nmse_db": ...}` schema as the real per-CR metrics, they silently overwrote
the correct entry in the results dict (last file processed, alphabetically, won). The comparison
table and NMSE-vs-CR plot briefly reported the QAT-fine-tuned-at-4-bits number as if it were the
full-precision result — a ~13 dB error for outdoor CR=1/4, caught by noticing the "official" NMSE
had changed without anyone retraining the CR sweep.

**Fix:** replaced the suffix-blocklist filter with an explicit regex requiring the whole filename
to match `{scenario}_cr[0-9.]+\.json` exactly — an allowlist, not a blocklist, so any *future*
sibling file pattern in the same metrics directory fails safe (excluded by default) instead of
needing to be remembered and added to an exclusion list each time.

### Bug 5 — PCA baseline wasn't actually reproducible

Found during an audit pass re-verifying every number in this README against the underlying data:
`sklearn.decomposition.PCA(n_components=m)` was called with no `random_state`. For large `m`
(specifically `m=512`, the CR=1/4 codeword size), sklearn's `svd_solver="auto"` heuristic picks a
*randomized* SVD solver instead of an exact one — reasonable for speed, but that solver is
nondeterministic without a fixed seed. Running the exact same PCA fit three times in a row on
identical data gave three different answers (−35.86, −35.88, −35.85 dB), a ~0.03 dB spread. For
smaller `m` (128, 64, 32) the `auto` heuristic picks a deterministic solver, so jitter there was
negligible (<0.005 dB).

This never changed a conclusion — re-running each comparison several times confirmed every
network-vs-PCA verdict in this README is robust to the jitter, including indoor's tightest margin
(CR=1/4, network wins by only 0.04 dB against PCA jitter of ~0.002 dB there) — but it meant the
exact PCA numbers reported weren't reproducible if someone re-ran the pipeline. **Fix:** added
`random_state=0` to the `PCA(...)` call in `pca_baseline.py`. All PCA-derived numbers in §4 were
regenerated against the now-deterministic version (values shifted by ≤0.05 dB from their original,
jitter-affected values — e.g. outdoor CR=1/4 PCA: −35.93 → −35.89 dB — matching the jitter band
observed above, not a real change).

## 4. Results

### NMSE vs. compression ratio (test set, best validation checkpoint)

| CR | M | indoor NMSE | outdoor NMSE |
|---|---|---|---|
| 1/4 | 512 | −2.12 dB | **−30.43 dB** |
| 1/16 | 128 | −0.53 dB | −10.66 dB |
| 1/32 | 64 | −0.30 dB | −4.94 dB |
| 1/64 | 32 | −0.16 dB | −2.05 dB |

Outdoor (sparser, easier to compress) beats indoor at every matched CR, as the physical premise
predicts. Indoor's gains are much more modest — consistent with its lower energy-capture ratio
(0.894 vs 0.958) meaning some information is already unrecoverable before the network even sees
the data.

*Figures:* `outputs/figures/nmse_vs_cr_{indoor,outdoor}.png`,
`outputs/figures/heatmap_comparison_{indoor,outdoor}.png` (true vs. reconstructed |H|, best CR).

### Neural network vs. PCA baseline

A linear PCA/SVD compressor at the *same* codeword size, fit on the training set, is included so
the network's gain isn't taken on faith.

| CR | indoor (NN vs PCA) | outdoor (NN vs PCA) |
|---|---|---|
| 1/4 | **−2.12** vs −2.08 | −30.43 vs **−35.89** (PCA wins) |
| 1/16 | **−0.53** vs −0.48 | **−10.66** vs −7.35 |
| 1/32 | **−0.30** vs −0.24 | **−4.94** vs −3.47 |
| 1/64 | **−0.16** vs −0.12 | **−2.05** vs −1.60 |

**This table went through two real revisions, both worth documenting** (PCA numbers below are as
they were observed at each historical stage, before the Bug 5 reproducibility fix — expect ~0.05 dB
drift from the current table above, not a real change). The first pass used the original paper's
single-`Conv2d(2→2)`-layer encoder, and PCA beat the network at CR=1/4 in *both* scenarios — badly
so for outdoor (−20.03 dB network vs −35.93 dB PCA). That was investigated
directly rather than shrugged off: training the CR=1/4 models 3× longer (600 vs 200 epochs) barely
moved the result (outdoor: −20.03 → −20.54 dB), which ruled out undertraining. The actual cause
was **encoder capacity** — a single conv layer followed directly by the linear bottleneck
projection gives the encoder almost no nonlinear feature-extraction depth, so it struggled to even
match what a *plain linear* method (PCA) could do once the codeword got large. Deepening the
encoder to three conv layers (`2→8→16→2`, mirroring the decoder's `RefineNetUnit` widths, see §2)
and retraining the whole sweep:
- **Closed the gap completely for indoor** — the network now wins at every CR, including 1/4.
- **Narrowed it substantially for outdoor** (15.9 dB gap → 7.7 dB gap at CR=1/4) and **widened the
  network's lead at every other outdoor CR** (e.g. 1/16's margin nearly quadrupled: 0.8 dB → 3.3 dB).
- **Left outdoor's CR=1/4 still losing to PCA** at −28.25 dB vs −35.93 dB.

The second revision specifically chased that remaining outdoor CR=1/4 gap, testing the two obvious
levers independently (both starting from the deeper-encoder result above, outdoor CR=1/4 only, 200
epochs each):
- **More training data** (30k samples instead of 10k, same architecture): **−30.43 dB** — the
  winner, and adopted as the official result above.
- **A wider encoder** (`2→16→32→2` instead of `2→8→16→2`, standard 10k data, exposed as
  `--encoder-width` on `scripts/train.py`, stored per-checkpoint so it round-trips through every
  downstream script automatically): −29.69 dB — real improvement, but smaller than more data alone.

Both levers helped independently (neither was tried combined, nor was more epochs re-tried on top
of either — diminishing-return territory by this point), and both still leave outdoor CR=1/4
losing to PCA, now by 5.5 dB instead of 7.7 dB. Both CR=0.25 runs hit their best checkpoint at
epoch 199/200 — still improving when training stopped, same pattern as every earlier attempt at
this specific operating point — so there is very likely more available with either more epochs or
combining both levers; not chased further here. The interpretation from the first revision stands:
outdoor's clustered generator produces channels with unusually strong *global* linear structure,
which is exactly the regime linear PCA is best at, and CR=1/4 has enough codeword budget for PCA to
exploit nearly all of it.

*Figures/data:* `outputs/figures/nmse_vs_cr_*.png` (both series plotted together),
`outputs/tables/comparison_{indoor,outdoor}.csv`.

### Quantization (stretch goal): how many bits does the codeword actually need?

Real feedback is a bitstring, not a continuous vector. Each already-trained codeword was
quantized post-hoc with a uniform scalar quantizer (range fixed from training-set statistics, as
a real system would) and decoded, to measure accuracy lost as bit-width shrinks — no retraining
required.

**Finding, consistent across both scenarios and all CRs (re-run against the deeper-encoder
checkpoints from §4):**
- Below **~3 bits/element**, reconstruction collapses (worse than the 0 dB trivial baseline).
- By **~6 bits/element**, quantized performance is within ~0.001–0.004 dB (indoor) or
  ~0.004–2.9 dB (outdoor) of full 32-bit float — effectively converged.
- Higher-fidelity codewords are *more* sensitive to quantization, not less, because quantization
  adds a roughly fixed noise floor that only matters once the network's own reconstruction error
  drops below it:
  - **Indoor** (unquantized NMSE only −0.16 to −2.1 dB to begin with) is already near its ceiling
    by **4 bits** (≤0.07 dB loss at every CR) — its low-precision reconstructions have nothing left
    for quantization noise to meaningfully add to.
  - **Outdoor's low CRs** (1/64, 1/32, 1/16 — unquantized −2.1 to −10.7 dB) are likewise within
    ~0.02–0.7 dB of their ceiling by 4 bits.
  - **Outdoor's CR=1/4** (now a much higher-fidelity **−28.25 dB** codeword, up from −20.03 dB with
    the old encoder) is the clear outlier, and the gap grew *with* the fidelity improvement: 4 bits
    now leaves an **11.7 dB** gap (was 6.8 dB), 5 bits leaves 6.4 dB, and it takes **6–8 bits** to
    close to within ~3 dB. The more accurately a codeword encodes the channel, the more bits it
    takes to transmit that accuracy losslessly — a genuinely intuitive result once you see it, but
    the wrong takeaway ("4 bits is always enough") would have been easy to draw by looking at only
    one scenario, and would have gotten *more* wrong, not less, as the network improved.

So the practically deployable answer is CR-dependent: **~4 bits/element** is enough everywhere
*except* for a codeword that's already achieving very high fidelity (like outdoor at CR=1/4),
which needs closer to **6–8 bits** to not throw that fidelity away.

*Figures/data:* `outputs/figures/nmse_vs_bits_{indoor,outdoor}.png`,
`outputs/metrics/{scenario}_cr{cr}_quant.json`.

### Quantization-aware fine-tuning (QAT): recovering the loss

Post-hoc quantization (above) quantizes a model that was never trained to expect it. The natural
follow-up: continue training each checkpoint with the codeword quantized *in the forward pass*
(via a straight-through estimator — forward pass quantizes, backward pass pretends it didn't, so a
training signal can still flow through the otherwise non-differentiable rounding step), at a fixed
target bit-width, so the decoder learns to compensate for that specific quantization noise. Fine-
tuned for 50 epochs at `lr=1e-3` (a lower rate than the `1e-2` used for training from scratch —
this is fine-tuning an already-good model, not learning from nothing; `1e-2` was tried and
destabilized it instead) across all 4 CRs, both scenarios.

**First pass covered `bits ∈ {3, 4}`** (reasoning at the time: below 3 bits there's likely too
little information left to recover, and by 6+ bits post-hoc is already near-lossless with nothing
to gain) — **finding: real, sometimes substantial, but partial recovery, narrows the gap without
closing it:**

| scenario | CR | bits | post-hoc | QAT | recovery |
|---|---|---|---|---|---|
| outdoor | 1/4 | 3 | −7.91 dB | **−11.18 dB** | +3.27 dB |
| outdoor | 1/4 | 4 | −14.37 dB | **−15.50 dB** | +1.13 dB |
| outdoor | 1/16 | 3 | −8.00 dB | **−9.41 dB** | +1.42 dB |
| outdoor | 1/32 | 3 | −4.16 dB | **−4.70 dB** | +0.54 dB |
| indoor | 1/4 | 3 | −1.76 dB | **−2.00 dB** | +0.23 dB |
| indoor | 1/64 | 4 | −0.15 dB | −0.15 dB | +0.01 dB (nothing left to recover) |

That "below 3 bits there's too little information to recover" assumption turned out to be wrong —
**the second pass tested `bits ∈ {1, 2}` anyway, and QAT recovers dramatically there, not
marginally:**

| scenario | CR | bits | post-hoc | QAT | recovery |
|---|---|---|---|---|---|
| outdoor | 1/4 | 1 | +12.36 dB | **−6.94 dB** | **+19.29 dB** |
| outdoor | 1/32 | 1 | +15.51 dB | **−2.94 dB** | +18.44 dB |
| outdoor | 1/16 | 1 | +13.36 dB | **−5.01 dB** | +18.36 dB |
| indoor | 1/4 | 1 | +14.10 dB | **−1.02 dB** | +15.12 dB |
| outdoor | 1/64 | 1 | +6.38 dB | **−1.20 dB** | +7.58 dB |
| outdoor | 1/4 | 2 | +0.30 dB | **−7.26 dB** | +7.56 dB |

At **1 bit per codeword element — a single sign bit, the most extreme quantization possible** —
post-hoc quantization is always catastrophic (+6 to +16 dB, far worse than the trivial 0 dB
baseline: the reconstruction is actively worse than feeding back nothing). QAT turns every one of
these into a genuinely *usable* system (−1 to −7 dB, comfortably beating the trivial baseline),
typically recovering 15–19 dB in the process. This is the strongest result in the whole
quantization investigation, and it directly contradicts the reasonable-sounding assumption that
motivated skipping 1–2 bits in the first pass — a good reminder to actually run the experiment
instead of reasoning your way out of it.

Across both passes, the pattern holds: **the worse post-hoc quantization was, the more QAT
recovers**, and cases already near their ceiling (indoor's low CRs, any scenario at bits ≥5) have
essentially nothing to gain. But recovery is never *complete* — even outdoor CR=1/4's best case
(+19.3 dB at 1 bit) lands at −6.9 dB, well short of the −30.43 dB unquantized ceiling.
Quantization noise is a real information-theoretic floor; QAT makes the model more *robust* to
that floor without making it disappear.

*Figures/data:* `outputs/figures/nmse_vs_bits_{indoor,outdoor}.png` (QAT points overlaid as star
markers on the post-hoc curves), `outputs/metrics/{scenario}_cr{cr}_b{bits}_qat.json`.

### The actual deployable question: total feedback bits vs. accuracy

Compression ratio and bit-width were swept separately above, but the real cost of feedback is
their product: `total_bits = M × bits/element`. A combined plot makes the actual trade space
visible — for a fixed feedback budget, which (CR, bits) combination gives the best accuracy? — with
a Pareto frontier (the lower envelope over every combination) marking which points are worth
choosing versus dominated by a cheaper alternative that does just as well.

Both scenarios show the same qualitative shape: **low bit-widths at any CR are always dominated**
(a coarsely-quantized large codeword is worse *and* more expensive than a well-quantized small
one), so the frontier tracks each CR's curve only once it's had enough bits to stop being
quantization-limited, then hands off to the next larger CR. Post-hoc 1-2 bit points are never on
the frontier in either scenario — they're the catastrophic failures from the QAT section above.
QAT visibly pulls several points onto (or much closer to) the frontier that weren't there before —
most strikingly outdoor's 1-bit points at CR=1/16 and CR=1/32, which land almost exactly on the
frontier despite their post-hoc versions being off the top of the chart entirely.

*Figures:* `outputs/figures/pareto_bits_vs_nmse_{indoor,outdoor}.png`.

### Hyperparameter sensitivity: is `lr=1e-2` actually the right choice?

The training learning rate (`1e-2`, see §3 Bug 3) was chosen because `1e-3` demonstrably failed to
converge in reasonable time — but "better than a broken value" isn't the same as "correct." Tested
`lr ∈ {5e-3, 1e-2, 2e-2}` at one representative CR (1/16, the "sweet spot") per scenario, 200
epochs each:

| scenario | lr=5e-3 | lr=1e-2 (current) | lr=2e-2 |
|---|---|---|---|
| indoor | −0.51 dB | −0.53 dB | −0.56 dB |
| outdoor | −10.64 dB | **−10.66 dB** | −9.60 dB |

**Finding: `1e-2` is already the right balanced choice, no change made.** Indoor is nearly flat
across all three values (2e-2 marginally better, by 0.03 dB — noise-level, and per the seed-variance
study below, indoor's run-to-run std at this CR is ~0.02 dB, so this single-seed 0.03 dB gap isn't
distinguishable from noise). Outdoor is far more sensitive: 2e-2 is a full 1.06 dB *worse*,
converging early (epoch 69/200) to a worse optimum, consistent with the instability seen when a
too-high LR was tried during QAT fine-tuning (§4
Quantization-aware fine-tuning). `5e-3` is safe but gives up a little accuracy on outdoor for no
benefit on indoor. `1e-2` is the value that's good on both scenarios simultaneously — validated,
not just inherited from "whatever fixed the bug."

### Seed variance: are the tight margins real?

Every result above (§4's tables) came from a single training run per (scenario, CR) — seed 0. Some
of the resulting network-vs-PCA margins are small (indoor's are all under 0.06 dB). Since neural
network training is stochastic (weight init, data shuffling order) and PCA is now deterministic
(§3 Bug 5), a natural question: are those margins a real effect, or could a different seed have
gone the other way? Retrained every (scenario, CR) combination at 2 more seeds (`scripts/train.py
--seed 1`, `--seed 2`, otherwise identical), giving 3 independent runs per cell, aggregated by
`scripts/seed_variance.py`:

| scenario | CR | seed 0 (official) | seed 1 | seed 2 | mean | std | PCA | worst-seed margin |
|---|---|---|---|---|---|---|---|---|
| indoor | 1/4 | −2.12 | −2.25 | −2.36 | −2.25 | 0.121 | −2.08 | **+0.04 dB** |
| indoor | 1/16 | −0.53 | −0.57 | −0.53 | −0.54 | 0.022 | −0.48 | +0.05 dB |
| indoor | 1/32 | −0.30 | −0.27 | −0.30 | −0.29 | 0.020 | −0.24 | +0.03 dB |
| indoor | 1/64 | −0.16 | −0.15 | −0.14 | −0.15 | 0.011 | −0.12 | +0.02 dB |
| outdoor | 1/4 | −30.43 | −31.07 | −30.35 | −30.62 | 0.394 | −35.89 | **−5.53 dB (PCA wins)** |
| outdoor | 1/16 | −10.66 | −9.35 | −10.25 | −10.09 | 0.674 | −7.35 | +1.99 dB |
| outdoor | 1/32 | −4.94 | −5.82 | −5.87 | −5.55 | 0.521 | −3.47 | +1.48 dB |
| outdoor | 1/64 | −2.05 | −3.48 | −3.83 | −3.12 | 0.940 | −1.61 | +0.45 dB |

("worst-seed margin" = PCA NMSE minus the *worst* (least-negative) of the 3 network seeds — i.e.
the margin in the least favorable of the 3 runs actually observed, not the mean.)

**Finding: every qualitative "network beats PCA" claim in this README holds in all 3 seeds tried
— including every one of indoor's tight margins — but not all of them are comfortable, and the
original seed-0-only numbers were sometimes misleading about the typical case, in both
directions:**
- **Indoor CR=1/4 is the least statistically robust of the "network wins" claims** — its std
  (0.121 dB) is *larger* than its own mean margin over PCA (0.162 dB, roughly 1.3 standard
  deviations), meaning a 95%-style confidence interval from just 3 samples would plausibly include
  zero. It won in all 3 trials run here, but this is the one claim in the whole README that a
  4th unlucky seed could most plausibly flip.
- **Indoor's other 3 CRs are comfortably robust** (margin-to-std ratios of ~2.6–3×) despite having
  *even smaller* raw margins (0.02–0.05 dB) than CR=1/4 — a small margin with low variance is more
  trustworthy than a larger margin with high variance, which is easy to get backwards by eyeballing
  the table in §4 alone.
- **Outdoor's CR=1/64 win is the least comfortable of outdoor's 3 "network wins"** (worst-seed
  margin +0.45 dB against a std of 0.94 dB) — real in every trial so far, but the least buffer of
  any outdoor CR.
- **Outdoor CR=1/4's "PCA wins" conclusion is not in doubt** — the ~5.5 dB gap dwarfs the observed
  0.39 dB network std by more than an order of magnitude; no plausible seed changes this one.
- The seed-0-only margins in §4's main tables weren't cherry-picked to look good: for indoor CR=1/4
  specifically, seed 0 was actually the *worst* of the 3 seeds tried (smallest margin over PCA),
  not the best.

PCA and the quantization/QAT sweeps remain single-seed/deterministic by construction (PCA has a
fixed `random_state`; quantization is a fixed arithmetic transform) — only the network *training*
itself is stochastic, and that's now what's covered here. The quantization-aware fine-tuning runs
(§4) were not re-run across seeds; whether those recoveries are similarly robust is untested.

*Figures/data:* error bars (±1 std) on `outputs/figures/nmse_vs_cr_{indoor,outdoor}.png`,
`outputs/tables/comparison_{indoor,outdoor}.csv` (`nn_std_db` column),
`outputs/metrics/{scenario}_cr{cr}_seeds.json`.

## 5. Project layout

```
csinet/
├── channel_model.py   synthetic clustered (Saleh-Valenzuela-style) multipath generator
├── transform.py       2D DFT → angular-delay domain, truncation, RMS-scaling for the network
├── dataset.py          split construction + torch Dataset
├── deepmimo_adapter.py real (DeepMIMO) channel ingestion + spatial-holdout split (see §8)
├── metrics.py          canonical NMSE(dB)
├── models/csinet.py    encoder / decoder / RefineNet
├── train.py             training loop (importable core)
├── pca_baseline.py     PCA/SVD baseline at matched compression ratio
├── quantize.py          uniform scalar codeword quantizer (numpy, post-hoc + torch STE for QAT)
├── qat.py                quantization-aware fine-tuning loop
└── viz.py                 all plots (matplotlib, static PNG)
scripts/                CLI entry points (see below), incl. generate_dataset_deepmimo.py (§8)
tests/                     pytest unit tests, one file per csinet/ module (38 tests)
outputs/                 gitignored: data/, checkpoints/, checkpoints_qat/, metrics/, figures/, tables/
deepmimo_scenarios/    gitignored: downloaded DeepMIMO ray-tracing scenario data (§8)
```

## 6. Running it

```bash
uv sync   # torch, numpy, scipy, matplotlib, scikit-learn, tqdm, pytest

uv run pytest tests/ -v

# 1. generate data (repeat per scenario)
uv run python scripts/generate_dataset.py --scenario indoor \
    --n-train 10000 --n-val 2000 --n-test 2000 --out-dir outputs/data
uv run python scripts/generate_dataset.py --scenario outdoor \
    --n-train 10000 --n-val 2000 --n-test 2000 --out-dir outputs/data

# 2. train the CR sweep (repeat per scenario). --encoder-width defaults to 8 16;
# pass e.g. --encoder-width 16 32 for the wider-encoder experiment (README §4)
uv run python scripts/train.py --scenario indoor \
    --cr 0.25 0.0625 0.03125 0.015625 --epochs 200 --lr 1e-2 \
    --data-dir outputs/data --out-dir outputs

# 3. PCA baseline (repeat per scenario)
uv run python scripts/run_pca_baseline.py --scenario indoor \
    --cr 0.25 0.0625 0.03125 0.015625 --data-dir outputs/data --out-dir outputs

# 4. quantization sweep (repeat per scenario)
uv run python scripts/quantize_eval.py --scenario indoor \
    --cr 0.25 0.0625 0.03125 0.015625 --bits 1 2 3 4 5 6 8 \
    --data-dir outputs/data --checkpoints-dir outputs/checkpoints --out-dir outputs

# 5. quantization-aware fine-tuning (repeat per scenario) -- run the full bits 1-4
# range; don't assume very low bit-widths aren't worth fine-tuning (README §4 -- QAT
# recovers 15-19 dB at 1 bit, the biggest gains in the whole quantization sweep)
uv run python scripts/qat_finetune.py --scenario indoor \
    --cr 0.25 0.0625 0.03125 0.015625 --bits 1 2 3 4 --epochs 50 --lr 1e-3 \
    --data-dir outputs/data --checkpoints-dir outputs/checkpoints --out-dir outputs

# 6. seed variance (optional, repeat per scenario) -- rerun the CR sweep at 2 more
# seeds, then aggregate; make_plots picks up the result automatically and adds error
# bars (README §4 "Seed variance")
uv run python scripts/train.py --scenario indoor --cr 0.25 0.0625 0.03125 0.015625 \
    --epochs 200 --lr 1e-2 --seed 1 --data-dir outputs/data --out-dir outputs/seeds/seed1
uv run python scripts/train.py --scenario indoor --cr 0.25 0.0625 0.03125 0.015625 \
    --epochs 200 --lr 1e-2 --seed 2 --data-dir outputs/data --out-dir outputs/seeds/seed2
uv run python scripts/seed_variance.py --scenario indoor \
    --cr 0.25 0.0625 0.03125 0.015625 --official-dir outputs \
    --seed-dirs outputs/seeds/seed1 outputs/seeds/seed2 --out-dir outputs

# 7. plots (repeat per scenario) -- make_quant_plots picks up QAT results automatically
# if present, and also produces the combined bits-vs-NMSE Pareto curve; make_plots
# picks up seed-variance results automatically (step 6) and adds error bars
uv run python scripts/make_plots.py --scenario indoor \
    --data-dir outputs/data --metrics-dir outputs/metrics --out-dir outputs
uv run python scripts/make_quant_plots.py --scenario indoor \
    --metrics-dir outputs/metrics --out-dir outputs
```

**Real data (DeepMIMO, see §8)** uses the same steps 2–7 above unchanged (just pass
`--scenario outdoor_real` or `--scenario indoor_real`) — only step 1 differs:

```bash
uv sync --group deepmimo   # optional group: deepmimo==4.0.0b7, pandas -- large, network-fetching

# 1. generate data from a real DeepMIMO scenario (repeat per scenario). Downloads the
# scenario's ray-tracing data on first use (multi-hundred-MB to a few GB, cached in
# deepmimo_scenarios/); uses a blocked spatial holdout split by default (§8) --
# --chunk-size controls peak memory (users processed per batch), lower it on a
# memory-constrained machine.
uv run python scripts/generate_dataset_deepmimo.py --scenario outdoor_real \
    --n-train 10000 --n-val 2000 --n-test 2000 --out-dir outputs/data
uv run python scripts/generate_dataset_deepmimo.py --scenario indoor_real \
    --n-train 10000 --n-val 2000 --n-test 2000 --out-dir outputs/data
```

## 7. Known limitations / possible improvements

- **COST2100 still scoped out** (the standard `.mat` files used by the original paper live behind a
  Dropbox folder link with no scrapeable listing — a blind multi-GB download wasn't judged worth
  the risk). DeepMIMO real-channel data was added instead (see §8) since it has a proper
  programmatic downloader; results are kept as a fully separate set of scenarios (`*_real`), never
  merged with the synthetic tables above.
- **`indoor_real`'s extreme compressibility is scenario-specific, not a general "indoor" claim.**
  It's a single small (2m×7m) room at 28GHz with near-universal line-of-sight — see §8. A different
  indoor scenario (larger, richer scattering, sub-6GHz) would likely look nothing like this.
- **PCA still beats the network at outdoor CR=1/4** (see §4) — deepening the encoder closed this
  gap for indoor and narrowed it for outdoor (15.9 dB → 7.7 dB), and a further round (more training
  data, 10k → 30k samples) narrowed it again to 5.5 dB, but didn't close it. Both of that second
  round's CR=1/4 runs hit their best checkpoint in the final training epoch — still improving when
  cut off — so more epochs, or combining more data with the (separately-tested, also-positive)
  wider encoder, is the next thing worth trying.
- **QAT recovers real accuracy but doesn't close the quantization gap** (see §4) — even the best
  case (outdoor CR=1/4 at 1 bit, +19.3 dB recovery) lands well short of the unquantized ceiling.
  Every bit-width 1–4 has now been fine-tuned; higher bit-widths (5+) weren't, since post-hoc
  quantization is already near-lossless there with nothing to recover.
- **Hyperparameter tuning was narrow, not absent.** The learning rate was validated with a 3-point
  sweep at one representative CR per scenario (§4) and confirmed near-optimal; epoch count and
  architecture width were each tested at exactly one alternative operating point (outdoor CR=1/4)
  rather than swept systematically across the whole grid.
- **Seed variance covers the CR sweep only** (§4 "Seed variance") — 3 seeds per (scenario, CR),
  confirming every network-vs-PCA verdict holds directionally but flagging indoor CR=1/4 and
  outdoor CR=1/64 as the least statistically comfortable wins. The QAT fine-tuning recoveries and
  the wider-encoder/more-data CR=1/4 experiments were each run at a single seed and not re-checked
  for variance — plausible next step if those specific numbers need to be load-bearing somewhere.

## 8. Real-channel data (DeepMIMO)

Everything above (§1–§7) uses the synthetic clustered-multipath generator. This section adds real,
ray-traced channel data as a **separate, additional** pair of scenarios — `outdoor_real` and
`indoor_real` — never merged with the synthetic `indoor`/`outdoor` tables above. The motivation was
two open findings from §4 that were explicitly attributed to the synthetic generator rather than to
anything fixable by more tuning: outdoor CR=1/4 losing to PCA by 5.5 dB, and indoor CR=1/4's
NN-beats-PCA margin (0.04 dB) being smaller than its own seed noise (0.121 dB). Real data is the one
change that can actually test whether those are true properties of CsiNet-style compression, or
artifacts of the synthetic model being too clean (or too irregular) in the wrong ways.

**Why DeepMIMO and not COST2100** (the original paper's dataset, see §7): COST2100's standard files
are distributed via a Dropbox folder with no programmatic listing. DeepMIMO (`pip install
deepmimo`) has an actual downloader API (`dm.download`/`dm.load`), which is what made it tractable
to integrate without a manual, unautomatable download step.

### Scenarios

| | DeepMIMO handle | environment | band | grid | candidate / active users |
|---|---|---|---|---|---|
| `outdoor_real` | `asu_campus_3p5` | ASU campus (outdoor) | 3.5 GHz | 1 m spacing, 410 m × 320 m | 131,931 / 85,157 |
| `indoor_real` | `i2_28b` | small room + hallway | 28 GHz (mmWave) | 1 cm spacing, 2 m × 7 m | 140,901 / 140,901 |

Both configured with a 32-element ULA at the base station and the full 1024-subcarrier OFDM
channel, matching the synthetic pipeline's `n_ant`/`n_subcarriers` conventions exactly — everything
downstream of channel generation (`to_angular_delay`, RMS normalization, the network, PCA,
quantization, QAT) is byte-for-byte the same code path as the synthetic scenarios, via
`csinet/deepmimo_adapter.py`'s `build_real_split`, which produces the identical `{"h":, "scale":}`
`.npz` contract as the synthetic `build_split`.

**Implementation notes, briefly:**
- `pip install deepmimo` resolves to the old stable v3 API by default; the auto-downloading v4 API
  used here needed pinning explicitly (`deepmimo==4.0.0b7`, an optional dependency group — see §6).
- `ChannelParameters.ofdm.selected_subcarriers` defaults to `[0]` only, separately from the
  `subcarriers` count field — easy to silently get a single-subcarrier "channel" instead of the
  full 1024-wide one if not set explicitly.
- Real channel power is physically tiny (~1e−17, genuine path loss) and varies by orders of
  magnitude with distance from the base station, unlike the synthetic generator which normalizes
  to unit energy at construction time. `normalize_unit_energy` replicates that convention
  per-sample so PCA/NMSE comparisons aren't dominated by trivial path-loss variation — its
  zero-power guard must use an exact-zero check (`power > 0`), not an absolute floor like `1e-12`;
  the latter was tried first and incorrectly discarded ~73% of legitimately-attenuated-but-valid
  users during testing.
- Computing the full 1024-subcarrier channel for the ~14,000 users needed for a 10k/2k/2k split all
  at once needs several GB of RAM and crashed the development machine once. Fixed by chunked
  processing (`build_real_split`, default `chunk_size=500`): compute, normalize, and truncate each
  chunk of users down to the small 32-tap representation before moving to the next chunk, so peak
  memory is bounded by chunk size, not total dataset size.

### Split methodology: two false starts and a fix

Getting a real, trustworthy train/val/test split out of a dense spatial grid of user positions
took three attempts — the first two produced headline numbers that were later shown to be
artifacts, caught the same way the §3 bugs were: by noticing a result was *too* good (or too bad)
to be believed, and testing why rather than reporting it.

**Attempt 1 — random shuffle split.** DeepMIMO users sit on a dense spatial grid (`i2_28b`'s is 1 cm
apart). A plain seeded permutation split routinely placed a "held-out" test point immediately next
to several training points — both PCA and the network then partly interpolate between
near-duplicate samples instead of generalizing. **Symptom:** `indoor_real`'s PCA baseline hit
**−120.03 dB at CR=1/4** — an implausible number for any real compression task. **Fix (partial, see
attempt 3):** replace the shuffle with a genuine spatial holdout.

**Attempt 2 — one contiguous spatial region.** Train/val/test occupying disjoint bands along one
coordinate axis (with a buffer gap) fixed the near-duplicate leakage, but for a geographically large
scenario the held-out band can be a statistically *different environment* — different buildings,
different distance/angle-to-base-station distribution — not just a different sample of the same
environment. **Symptom:** `outdoor_real` (a 410 m × 320 m campus split into one contiguous ~40%
test region) **collapsed to ~0 dB for both PCA and the network** — not a network failure, since PCA
(a non-overfitting linear method) failed identically, which is what made this a domain-shift
diagnosis rather than a training problem.

**Attempt 3 — many small, scattered, buffered blocks (the fix).** Tile the area into a grid of small
blocks; a random fraction become held-out (val/test), every block *adjacent* to a held-out block is
excluded as a buffer, and the rest is train. Held-out data is now scattered across the same overall
geographic diversity as training (same buildings, same distance distribution) while still
guaranteeing no near-duplicate point crosses the split — fixing attempt 2's domain shift without
reintroducing attempt 1's leakage. One additional tuning step was needed: the campus's active users
are sparse/irregular (65% active, likely following streets/walkways rather than uniformly filling
the bounding box), so the default `holdout_fraction=0.3` left the buffer removal eating most of the
train pool (1,165 of 10,200 requested) — lowering to `holdout_fraction=0.2` reliably restored full
counts on both scenarios.

**The fix's own validation, and the most interesting finding of this whole section:** re-running
`indoor_real`'s PCA baseline under the corrected split gave **−119.99 dB — virtually unchanged from
the leaky attempt-1 number.** That's not a bug surviving the fix; it's evidence the original number
was never really about leakage for this scenario. A 2 m × 7 m room with near-universal line-of-sight
at 28 mmWave varies so smoothly and predictably with position that it's close to a genuine
low-dimensional linear manifold *across the entire room*, buffered split or not — PCA (the
mathematically optimal linear compressor) captures that almost perfectly regardless of exactly which
points end up in which split. `outdoor_real`, by contrast, changed dramatically at every stage
(−120 → ~0 → real numbers), because its issue actually was a data-splitting artifact, not a genuine
property of the scenario. Two scenarios, two different diagnoses, and the fix distinguished between
them correctly instead of applying one story to both.

### Results: NMSE vs. compression ratio, network vs. PCA

| CR | `outdoor_real` NN (mean ± std) | `outdoor_real` PCA | `indoor_real` NN (mean ± std) | `indoor_real` PCA |
|---|---|---|---|---|
| 1/4 | −1.12 ± 0.17 dB | **−2.03 dB** (PCA wins) | −33.02 ± 0.79 dB | **−119.99 dB** (PCA wins) |
| 1/16 | **−0.81 ± 0.08 dB** | −0.44 dB | −32.49 ± 1.07 dB | **−64.79 dB** (PCA wins) |
| 1/32 | **−0.60 ± 0.03 dB** | −0.10 dB | −29.44 ± 0.32 dB | **−32.99 dB** (PCA wins) |
| 1/64 | **−0.31 ± 0.08 dB** | −0.03 dB | **−20.84 ± 0.59 dB** | −19.89 dB |

**`outdoor_real` gives modest, believable numbers** — nothing close to either failed-split extreme —
and notably **agrees in direction with the synthetic outdoor finding**: PCA wins at CR=1/4 in both,
just at very different magnitude (0.93 dB here vs. 5.5 dB synthetic). That's real support for the
synthetic pipeline's interpretation (outdoor-style sparse propagation has strong global linear
structure, which is exactly the regime linear PCA is best at) rather than a synthetic-generator
artifact — though the much smaller real-world gap suggests the synthetic outdoor scenario
overstates how *linear* real outdoor propagation actually is.

**`indoor_real` gives an extreme, but — per the split-validation above — believable result**: PCA
dominates by a wide margin at every CR except the smallest, where the network barely wins. This
doesn't validate or contradict the synthetic indoor finding in either direction (they're different
environments, different bands, different physical regimes) — it's better read as an independent data
point showing that *some* real, physically-grounded scenarios really are this close to linear, which
the deliberately clustered/irregular synthetic model doesn't capture at all.

*Figures/data:* `outputs/figures/nmse_vs_cr_{outdoor_real,indoor_real}.png`,
`outputs/figures/heatmap_comparison_{outdoor_real,indoor_real}.png`,
`outputs/tables/comparison_{outdoor_real,indoor_real}.csv`.

### Quantization and QAT on real data

Same post-hoc quantization sweep and QAT fine-tuning procedure as §4, unmodified, run against the
real checkpoints. The qualitative pattern from the synthetic pipeline holds exactly: catastrophic
below ~3 bits/element (worse than the 0 dB trivial baseline), and QAT recovers substantially —
**indoor_real's best recovery, +23.1 dB at CR=1/4/1-bit (post-hoc +9.06 dB → QAT −14.04 dB), is the
single largest recovery observed anywhere in this project, synthetic included (previous record:
+19.3 dB, synthetic outdoor).**

| scenario | CR | bits | post-hoc | QAT | recovery |
|---|---|---|---|---|---|
| outdoor_real | 1/16 | 1 | +13.61 dB | **−0.36 dB** | +13.97 dB |
| outdoor_real | 1/32 | 1 | +13.87 dB | **+0.06 dB** | +13.81 dB |
| outdoor_real | 1/4 | 1 | +9.91 dB | **−0.89 dB** | +10.81 dB |
| indoor_real | 1/4 | 1 | +9.06 dB | **−14.04 dB** | **+23.10 dB** |
| indoor_real | 1/16 | 1 | +9.34 dB | **−10.14 dB** | +19.48 dB |
| indoor_real | 1/64 | 1 | +12.61 dB | **−6.65 dB** | +19.26 dB |
| indoor_real | 1/32 | 1 | +10.65 dB | **−8.28 dB** | +18.93 dB |

As with synthetic data, recovery is never complete — even indoor_real's best case (−14.04 dB at
1 bit) lands well short of its −33.02 dB unquantized ceiling. Full per-bit-width (1–4) results:
`outputs/metrics/{outdoor_real,indoor_real}_cr{cr}_b{bits}_qat.json`,
`outputs/figures/nmse_vs_bits_{outdoor_real,indoor_real}.png`,
`outputs/figures/pareto_bits_vs_nmse_{outdoor_real,indoor_real}.png`.

### Seed variance on real data

Same procedure as §4: 2 additional seeds per (scenario, CR), aggregated with `seed_variance.py`.
Both real scenarios turn out to have exactly one statistically shaky margin each — a direct parallel
to the synthetic pipeline's indoor CR=1/4 finding, but at a different operating point in each case:

- **`outdoor_real`: every margin is robust.** Even the smallest, CR=1/16 (network wins by 0.31 dB,
  std 0.08 dB), comfortably clears its own noise.
- **`indoor_real`'s CR=1/64 is the one shaky claim**: the network's 0.32 dB win over PCA is smaller
  than its own seed std (0.59 dB) — a 4th seed could plausibly flip it. Every other `indoor_real` CR
  is robust by a wide margin (PCA's smallest win, CR=1/32's 3.9 dB, dwarfs its 0.32 dB std by 10×).

Unlike synthetic indoor (where the shaky margin was at the *largest* CR, 1/4), here it's at the
*smallest* CR — a reminder that "which operating point is statistically fragile" isn't something
that transfers from one scenario to another; it has to be checked per scenario.

*Data:* `outputs/metrics/{outdoor_real,indoor_real}_cr{cr}_seeds.json`, `nn_std_db` column in the
comparison tables above.

### Citation and license note

DeepMIMO scenario data (ray-traced from Remcom Wireless InSite) is distributed for research use;
if any result here is published or redistributed, cite the DeepMIMO paper (Alkhateeb, *"DeepMIMO: A
Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications,"* 2019) per its
terms. Downloaded scenario data and everything derived from it (`.npz` splits, checkpoints) stay
out of git (`deepmimo_scenarios/` and `outputs/` are both gitignored).
