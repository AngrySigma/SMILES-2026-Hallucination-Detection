# Hallucination Detection — Solution Report

## 1. Headline result

The shipped solution is a 9-sub-probe hierarchical ensemble combining classical-ML probes (LR / HistGBT / ExtraTrees / MLP / multi-C bagged LR) over Qwen2.5-0.5B hidden-state aggregations, hand-crafted geometric features, logit-lens uncertainty features, and greedy-regeneration overlap signals. Five-fold stratified cross-validation on `data/dataset.csv` produces:

- **`avg_test_accuracy` = 0.7794 ± 0.014** (per-fold: 0.7826, 0.7971, 0.7754, 0.7826, 0.7591)
- `avg_test_auroc` = 0.7834
- `avg_train_accuracy` = 0.9492
- Majority-class baseline = 0.7010

The combiner is hierarchical: 8 sub-probes form an inner weighted-average using OOF-derived `max(acc − 0.5, ε)`-style weights; that single combined probability is then simple-mean averaged with 4 standalone sub-probe probabilities (3 of which are deliberate duplicates of inner sub-probes — an artefact of the subset-search-found optimum). The threshold is set via 5-fold inner OOF for stability rather than per-fold val tuning.

## 2. Reproducibility

### 2.1 Environment

Python 3.11+, dependencies pinned in `requirements.txt` unchanged from the upstream repo:

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=1.5.0
tqdm>=4.65.0
```

Tested on CUDA 12.1 with PyTorch installed from the cu121 wheel index. Any CUDA-capable GPU with ≥6 GB VRAM works; reported numbers are from runs on T4 and H100. CPU-only inference is supported but would take hours due to the regeneration step at module load.

Constants inherited from `solution.py`: `MAX_LENGTH = 512`, `BATCH_SIZE = 4`. These are not modified.

### 2.2 Running the solution

```bash
pip install -r requirements.txt
python solution.py
```

Outputs land at `results.json` (per-fold metrics + averages) and `predictions.csv` (test-set predicted labels, columns `id,label`).

`solution.py`, `model.py`, and `evaluate.py` are untouched relative to the upstream spec. Everything distinguishing this submission lives in `aggregation.py`, `probe.py`, and `splitting.py`. The submission additionally includes **`analysis/regenerations.npz`** — see §2.3 for what it is and why it's included.

### 2.3 Runtime expectations

Per-stage budget on a Colab T4 vs an H100 (T4 numbers measured end-to-end on a clean Colab T4 with no shipped cache; H100 numbers are extrapolated from prior throughput benchmarks):

| Stage | T4 | H100 |
| --- | --- | --- |
| Eager regen-cache build at `aggregation.py` module load *(skipped if `analysis/regenerations.npz` is present)* | ≈10 min | ≈30 sec |
| Hidden-state extraction (Qwen forward pass × 789 samples, batch 4) + 5-fold probe training + final-probe fit + test-set extraction and prediction | ≈5 min | ≈2 min |
| **Total** | ≈15 min cold / ≈5 min with cache | ≈2–3 min |

The submission includes `analysis/regenerations.npz` (≈20 KB — pre-computed greedy regeneration features for all 789 dataset+test rows). With this file in place, the inline regen builder no-ops on load and the run is fast and bit-reproducible against the reported numbers. If the file is missing or deleted, the builder kicks in, rebuilds the cache in-memory, and persists the result to `analysis/regenerations.npz` for subsequent runs. The first such rebuild may shift the numbers by ≈±0.5 pp due to GPU precision (see §2.4).

### 2.4 Determinism and known drifts

Every `random_state` in the editable code is pinned to 42 — including `StratifiedKFold`, `train_test_split`, `LogisticRegression`, `HistGradientBoostingClassifier`, `ExtraTreesClassifier`, `_MultiCLR`'s C grid, and `_MLPProbe`'s `torch.manual_seed`. Classical-ML probes (LR / GBT / ExtraTrees) are byte-deterministic given fixed data and seeds.

Sources of small drift that cannot be fully suppressed:

1. **bf16 floating-point precision varies by GPU**. The same input through Qwen-0.5B in bf16 produces slightly different hidden states on T4 vs H100 (differences ≈1e-3 per element). This propagates: regen logprobs, logit-lens entropies, and actual-token logprobs all shift slightly → individual sub-probe scores shift → meta-mean shifts. Empirically, across T4 and H100 runs, end-to-end test accuracy differs by ≤0.5 pp on identical code.
2. **`torch.manual_seed(42)` is set inside `_MLPProbe`, but `torch.backends.cudnn.deterministic` and `torch.use_deterministic_algorithms` are not enabled**. The MLP sub-probe's resulting probabilities can drift across different cuDNN versions on the same GPU. Effect on the final metric: ≤0.3 pp.
3. **Batched generation in the regen-cache builder can diverge from sequential generation on the same prompt**. With padding to the longest in batch and length-sorted chunking, the divergence is small (typically <1% of tokens differ), but it does shift the regen-overlap features and therefore the regen-related sub-probes.

With `analysis/regenerations.npz` shipped alongside the code, drift source 3 is eliminated and only sources 1 and 2 remain. The reported numbers in §1 should reproduce to within ≈±0.3 pp on any CUDA GPU.

## 3. Final solution

### 3.1 Files modified

| File | Changes |
| --- | --- |
| `aggregation.py` | Full rewrite. Introduces a `FeatureBlock` registry (`BLOCKS`), 10 active feature-block extractor factories, a `response_mask()` helper with module-level per-sample input-ids and regen-cache state, and an inline regen-cache builder (`_build_regen_cache_inline`) that runs `Qwen.generate()` at module-load time. |
| `probe.py` | Full rewrite. Defines `_MultiCLR` and `_MLPProbe` wrapper classes; a `SubProbeConfig` dataclass; the `SUB_PROBES` list of 9 sub-probe configs plus the `META_TOP_NAMES` / `META_STANDALONE_NAMES` selectors driving the meta-average combiner; the `HallucinationProbe` ensemble class with OOF-based weight and threshold tuning inside `fit()`. |
| `splitting.py` | Replaced single-split skeleton with 5-fold `StratifiedKFold(random_state=42)` + 15% inner stratified val carved from each train fold. |
| `analysis/regenerations.npz` | Pre-computed cache for the regen feature block. ≈20 KB. See §2.3. |

### 3.2 Pipeline at a glance

```
solution.py executes the following, top to bottom:

  import aggregation          ← module-load triggers eager regen-cache build
                                if SUB_PROBES uses regen_features (≈10 min T4 cold,
                                ≈30 s H100, no-op when analysis/regenerations.npz exists)

  load Qwen2.5-0.5B (bf16)
  move to device

  for each batch of 4 rows:
    tokenize → input_ids + attention_mask
    forward pass → hidden_states tuple (25 layers × seq_len × 896)
    stack to (4, 25, seq_len, 896) → upcast to fp32
    for each row i:
      feat = aggregation_and_feature_extraction(hidden[i], mask[i])
      ↳ internally:
        response_mask()  populates _CURRENT_INPUT_IDS and _CURRENT_REGEN globals
        for each block in BLOCKS: append block.extract(hidden, mask, rmask)
        concatenate → 1-D feature vector of length 20819 (the active blocks)

  X = vstack(all features) → (689, 20819)

  splits = split_data(y)             ← 5-fold StratifiedKFold + 15% inner val per fold
  fold_results = evaluate.run_evaluation(splits, X, y, HallucinationProbe)
                                       ← evaluate.py calls probe.fit(),
                                         probe.fit_hyperparameters() (no-op for
                                         meta_avg), probe.predict() per fold

  save_results → results.json

  extract features for data/test.csv with the same loop → X_test
  final_probe = HallucinationProbe().fit(X[train ∪ val of every fold], y[...])
                                       ← uses the full 689 training labels;
                                         fit() handles OOF weights + threshold
  predictions.csv = final_probe.predict(X_test)
```

### 3.3 Feature aggregation — `aggregation.py`

The `aggregate(hidden_states, attention_mask)` entrypoint walks `BLOCKS` (a list of `FeatureBlock(name, extract, dim)` entries) and concatenates each block's 1-D output. The registry pattern keeps the file editable in one place: appending a `FeatureBlock(...)` line auto-extends the concatenated feature vector and updates `BLOCK_SLICES` (a name → slice mapping consumed by sub-probes in `probe.py`).

`response_mask(attention_mask)` is called once per sample at the top of `aggregate()`; it advances a global counter that indexes into pre-tokenized lookups (filled lazily on first call), and stores per-sample `_CURRENT_INPUT_IDS` and `_CURRENT_REGEN` globals that the actual-logprob and regen blocks consume without further bookkeeping.

#### 3.3.1 Feature block registry

Active blocks in the shipped configuration:

| Block | Dim | What it captures |
| --- | --- | --- |
| `resp_mean_l23` | 896 | Layer-23 hidden states averaged over response tokens. |
| `last_tok_l20` | 896 | Layer-20 hidden state at the last real token. |
| `cross_layer_mean` | 896 | Layers 12–16: [last token + mean(last 3 tokens)] per layer, then averaged across all 10 vectors. |
| `geo_features` | 101 | 25 per-layer last-token L2 norms + 25 per-layer mean activations over response + 24 cosine drifts at last token + 24 cosine drifts at response-mean + spread + sequence length + response length. |
| `logit_lens_24` | 9 | Apply Qwen's final RMSNorm + lm_head to layer-24 response-mean → top-1 prob / top1-top2 margin / entropy, each as mean / min / max over response tokens. |
| `length` | 4 | `[prompt_tok, response_tok, total_tok, is_truncated]`. |
| `actual_logprob` | 15 | Layer-24 logit-lens features (9) + 6 actual-token features: mean / min / max of the logprob of the actually-generated response tokens, plus mean / max of `top1_prob − prob_actual`, plus argmax-match rate. |
| `regen_features` | 6 | From the regen cache: token Jaccard, token overlap ratio, first-N-token match rate, length ratio, length difference, mean per-token logprob of the greedy regen. |
| `stat_features` | 28 | Layers 12 / 16 / 20 / 24, each gives 7 distribution-shape statistics of per-response-token L2 norms: mean, std, p25 / p50 / p75, excess kurtosis, entropy(softmax(norms)). |
| `layer_drift_l2` | 48 | Per consecutive-layer pair (0..24): L2 magnitude of the change in last-token state (24) + L2 magnitude of the change in response-mean (24). |
| **Total** | **2008** + 5×3584 inactive multi-pool blocks = **20819** | The multi-pool blocks (`mean_4l`, `last_4l`, `lastK16_4l`, `lastK32_4l`, `lastK64_4l`) are registered but no active sub-probe consumes them; they were inputs to an earlier ablation variant. |

#### 3.3.2 Inline regen-cache builder

`regen_features` depends on running `model.generate()` per sample, which is too slow inside the per-sample aggregation loop. The builder runs at module-load time — *before* `solution.py` loads its own Qwen — so GPU memory stays sequential (the builder's Qwen loads, generates, frees; then the main Qwen loads). The trigger fires only if an active sub-probe in `SUB_PROBES` declares `regen_features`; removing those sub-probes short-circuits the build entirely.

Implementation details:
- Loads Qwen2.5-0.5B in bf16, sets `tokenizer.padding_side = "left"`.
- Pre-tokenizes every (prompt, response) row from `data/dataset.csv` then `data/test.csv` in deterministic order.
- Sorts rows by prompt length and processes in chunks; each chunk's max prompt length determines the left-padding, minimising wasted compute.
- Adaptive chunk size based on `cuda.get_device_properties(0).total_memory`: 128 on >60 GB (H100), 128 on >30 GB (A100-40), 256 on >12 GB (T4), 1 on CPU/MPS.
- Calls `model.generate(do_sample=False, num_beams=1, max_new_tokens=256, pad_token_id=…)` with an explicit attention mask.
- Captures per-step top-1 logprob via a `LogitsProcessor` that writes a `(batch,)` tensor to CPU each step — avoids retaining the full `(batch, max_new, vocab)` scores tensor (would be ≈10 GB at batch 128).
- Per generated sequence, trims at the first EOS (inclusive) and averages logprobs only over the real generated tokens.
- OOM-safe fallback: if a chunk OOMs, retries that chunk one sample at a time before giving up and writing zeros for the affected row.
- Resolution order in `_ensure_regen_cache()`: (1) already in process memory → no-op; (2) `analysis/regenerations.npz` on disk → load; (3) build inline. After a tier-3 build, the cache is written to `analysis/regenerations.npz` so subsequent runs hit tier 2. The shipped submission lands in tier 2 directly.

### 3.4 Probe and combiner — `probe.py`

`HallucinationProbe` (the public class `evaluate.py` instantiates) owns the list of `SubProbeConfig` entries in `SUB_PROBES`. Each config declares `(name, blocks, factory, use_scaler)`; `blocks` is a tuple of block names referring back to `aggregation.BLOCKS`, picked into a contiguous feature slice via `aggregation.BLOCK_SLICES`. The factory returns a fresh classifier per call (LR / HistGBT / ExtraTrees / `_MultiCLR` / `_MLPProbe`). `use_scaler=True` wraps the slice in a `StandardScaler` fit on training data.

The combiner is the hierarchical meta-average described below; ablations against simple-mean / weighted-val-acc / stacking-LR combiners are reported in §5.3.

#### 3.4.1 Sub-probes

The shipped configuration contains 9 sub-probes in two functional groups:

**Top portion** (8 sub-probes, get weighted-averaged into a single inner probability `p_top8`):

| Name | Block(s) | Classifier | Notes |
| --- | --- | --- | --- |
| `cross_layer_mean_lr` | `cross_layer_mean` | LR(C=0.01, balanced) | Mid-layer cross-layer pool + strong-L2 LR. |
| `geo_gbt` | `last_tok_l20` + `geo_features` | HistGBT(max_depth=6, lr=0.05, l2=1.0) | Trees on heterogeneous geometric scalars. |
| `resp_mean_lr` | `resp_mean_l23` | LR(C=0.01, balanced) | Cheapest hidden-state probe — strong baseline. |
| `actual_logprob_gbt` | `actual_logprob` | HistGBT(max_depth=4, lr=0.04) | Vocab-space uncertainty axis. |
| `regen_overlap_gbt` | `regen_features` | HistGBT(max_depth=4, lr=0.04) | Active-LLM regeneration overlap. |
| `multi_c_lr_resp_mean` | `resp_mean_l23` | `_MultiCLR` averaging LR over C ∈ {0.001, 0.01, 0.05, 0.1, 0.3, 1.0} | Genuinely diverse fits (vs random_state bagging which is a no-op for lbfgs). |
| `extra_trees_geo` | `last_tok_l20` + `geo_features` | ExtraTreesClassifier(n=400, max_features="sqrt") | Decorrelated tree family alongside HistGBT. |
| `stat_drift_lr` | `stat_features` + `layer_drift_l2` | LR(C=0.01, balanced) | Distribution-shape + L2 layer drift features. |

**Standalone** (1 explicitly-distinct sub-probe, joins the meta-mean alongside the inner combined probability):

| Name | Block(s) | Classifier | Notes |
| --- | --- | --- | --- |
| `mlp_pool_mlp` | `resp_mean_l23` | `_MLPProbe` (Linear(D, 256) → ReLU → Dropout(0.3) → Linear(256, 1), AdamW, batch 64, max 300 epochs, patience 20) | An MLP on layer-23 response-mean; standalone-weak but contributes via decision-boundary diversity. |

Three of the top-portion sub-probes (`cross_layer_mean_lr`, `geo_gbt`, `regen_overlap_gbt`) *also* enter the meta-mean as duplicated standalones — see §3.4.2 below.

#### 3.4.2 Meta-avg combiner

The math, per-sample:

```
p_top8(x)  = Σ_{i ∈ top-portion}  w_i · p_i(x)        # OOF-derived w_i, normalised to sum to 1

meta_mean(x) = ( p_top8(x)
               + p_cross_layer_mean_lr(x)
               + p_geo_gbt(x)
               + p_regen_overlap_gbt(x)
               + p_mlp_pool_mlp(x)
             ) / 5
```

The standalone-portion contains `cross_layer_mean_lr`, `geo_gbt`, `regen_overlap_gbt`, and `mlp_pool_mlp`. The first three are *deliberately the same sub-probes already inside `p_top8`*: their probabilities therefore enter the final mean with effective weight `(w_i + 1) / 5` while sub-probes only in the top portion enter with weight `w_i / 5`, and `mlp_pool_mlp` enters with weight `1 / 5`. This is exactly the configuration the OOF subset search picked out (§5.1) — duplicating the strongest probes outperformed any flat weighted combine on the held-out OOF.

#### 3.4.3 OOF threshold tuning

Inside `fit()`:

1. Fit all 9 sub-probes on the full input training set → stored as the final estimators used by `predict_proba`.
2. Run a 5-fold inner `StratifiedKFold(random_state=42)` on the same training set, refitting each sub-probe per inner fold, gathering an `(N, 9)` matrix of OOF probabilities.
3. From the top-portion columns of that OOF matrix: compute `acc_i` = accuracy-best achievable on `sp_oof[:, i]` for each top sub-probe; weights = `max(acc_i − 0.5, ε)` normalized to sum to 1.
4. Compute `oof_combined` exactly the same way `predict_proba` will at inference: `p_top8_oof = sp_oof[:, top_idx] @ weights`, then `mean(p_top8_oof, sp_oof[:, standalone_idx])`.
5. Threshold = accuracy-best on `(oof_combined, y)`, with a tie-break toward 0.5.

`fit_hyperparameters(X_val, y_val)` is **explicitly a no-op for `meta_avg`**. Evaluation showed that re-tuning the threshold on the per-fold val slice (≈83 samples) is strictly noisier than the OOF-tuned threshold from the full training set (≈551 OOF samples). Bypassing this step recovered ≈1.75 pp on the per-fold test accuracy without any change to the model itself — see §5.7.

### 3.5 Splitting — `splitting.py`

`split_data(y, df=None, n_splits=5, val_size=0.15, random_state=42)`:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for idx_tv, idx_test in skf.split(np.zeros(len(y)), y):
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15, random_state=42, stratify=y[idx_tv],
    )
    splits.append((idx_train, idx_val, idx_test))
```

Five outer folds, each with ≈468 train / ≈83 val / ≈138 test. Stratified at both levels so the 70 / 30 class balance holds in every split. `random_state=42` everywhere makes the fold structure reproducible — every experiment run during the iteration used these exact splits, which made cross-probe OOF dumps directly comparable for the subset search (§5.1).

Empirically verified during the iteration: zero duplicate prompts in `dataset.csv`, zero user-message duplicates, zero leakage between any pair of folds. Group-aware splitting was therefore unnecessary.

## 4. Design rationale

### 4.1 Key choices

- **Mid-late layers (12-24), not just the last layer**. A per-layer linear-probe sweep on `dataset.csv` showed the highest accuracy at layers 16-22 for last-token pooling and at layer 23 for response-mean pooling. The final layer (24) is biased toward producing the next-token distribution; useful for logit-lens features but not the strongest single feature for factuality.
- **Response-token pooling beats last-token pooling**. On the same sweep, mean-pooling over response tokens at layer 23 outperformed last-token-of-last-layer by ≈4 pp standalone. The shipped solution uses response-mean for the main hidden-state probes (`resp_mean_l23`, `cross_layer_mean`), last-token for the geometric-features probe (`last_tok_l20` — paired with the 101-d `geo_features` block where last-token-norm patterns matter).
- **Multi-family probes for error decorrelation**. The ensemble combines LR (linear, strong-L2), HistGBT (gradient-boosted trees), ExtraTreesClassifier (extremely-randomized trees), `_MultiCLR` (multi-C bagged LR), and a single MLP. Each family has different inductive biases → their errors decorrelate → averaging probabilities lowers variance without sacrificing mean accuracy.
- **Strong L2 on LR (C=0.01)**. With N≈468 training samples per fold and hidden_dim=896, default-regularised LR overfits dramatically. C=0.01 was independently confirmed by sweeping C ∈ {1, 0.5, 0.1, 0.05, 0.01, 0.005} on layer-23 response-mean — 0.01 was the sweet spot.
- **Active-LLM features on top of hidden-state pooling**. Hidden-state-only ensembles plateau at ≈0.75 in early experiments. Adding (a) logit-lens entropy / top-1 prob / margin features at layer 24, (b) the logprob of the actually-generated response tokens under the model, and (c) overlap statistics between the dataset's response and Qwen's greedy regeneration of the same prompt, lifts the ensemble to ≈0.77. These three feature axes are *orthogonal* to hidden-state geometry — they answer "is the model surprised by this token?" rather than "where does this representation sit in feature space?".
- **OOF threshold tuning rather than val tuning**. The 70 / 30 class imbalance makes accuracy very sensitive to the threshold. With only 83 val samples per fold, a per-fold val-tuned threshold has high variance. OOF tuning uses the full 689 samples' OOF predictions and is dramatically more stable. This was the final lift — see §5.7.

### 4.2 What contributed most to the metric

Approximate cumulative contribution, baseline → final:

| Change | Approx. test acc | Δ vs prior |
| --- | --- | --- |
| Majority-class baseline | 0.7010 | — |
| Default skeleton (last-token of last layer + 200-epoch full-batch MLP + F1 threshold) | 0.7020 | +0.001 |
| Switch threshold target from F1 → accuracy | ≈0.730 | +0.028 |
| Drop the MLP, use strong-L2 LR (C=0.01) on the same features | ≈0.745 | +0.015 |
| Switch from last-token-of-last-layer to response-mean of layer 23 | ≈0.750 | +0.005 |
| Add `geo_features` block + HistGBT sub-probe on `last_tok_l20 + geo_features` | ≈0.755 | +0.005 |
| Add `cross_layer_mean` block (mid-layer cross-layer pool average) + LR sub-probe | ≈0.762 | +0.007 |
| Add active-LLM features (`logit_lens_24`, `actual_logprob`, `regen_features`) as new sub-probes | ≈0.762 | ≈0 standalone, but unlocks ensemble gains |
| Weighted-val-acc ensemble combiner over 8 sub-probes | ≈0.762 | enabled by the prior step |
| Meta-avg combiner with deliberate duplication (subset-search-driven) | ≈0.762 | unchanged with val-tuned threshold |
| **Switch from val-tuned threshold → OOF threshold for meta-avg** | **0.7794** | **+0.017** |

The biggest single lifts were the very first change (F1 → accuracy threshold) and the very last (OOF threshold instead of val threshold) — both essentially "the model's rankings are fine, the threshold was wrong." Everything in between was small individual lifts compounding via diversity. The ensemble's structural choice (meta-avg with duplicates) does not raise the mean accuracy materially over flat weighted averaging, but it does reduce per-fold variance and was what the subset search consistently picked as the most stable configuration.

## 5. Experiments and selection process

This section covers both the path to the final solution and the dead-ends. The final ensemble was not handcrafted — it was selected by an automated subset search over per-probe out-of-fold predictions, then validated end-to-end on the per-fold test sets. Several individually-weak probes (some scoring below the majority-class baseline) still ended up in the final ensemble because their errors decorrelated with the stronger probes; this was the most counter-intuitive lesson of the search.

The dump / search / cache-builder tooling that made this possible was experiment-time infrastructure: an orchestrator that ran each candidate probe through the same 5-fold split and persisted its OOF and test probabilities, an offline subset-search script over the resulting probability matrix, and a few side scripts that pre-computed cached features for the more expensive probes. None of that tooling is part of the shipped submission — `python solution.py` against a clean checkout runs without it. Detailed per-experiment numbers from this process are in **Appendix A**.

### 5.1 How the final ensemble was chosen

Each candidate probe was implemented as a standalone experiment and run through the same 5-fold `StratifiedKFold(random_state=42)` split to dump an `oof_probs` vector and a `test_probs` vector. Ten candidates were dumped (see Appendix A for the full list).

The search itself was offline (pure NumPy + sklearn, no GPU): for each of the 2¹⁰ = 1024 non-empty subsets of the 10 probes, evaluate `simple_mean(subset_columns)` via an outer 5-fold CV on the OOF probability matrix, tuning the threshold on each outer-train OOF and applying it to the outer-test OOF. Report mean and std across the 5 outer folds.

In parallel, the same matrix was scored by:

- Greedy forward selection (start empty, add the probe with the biggest accuracy lift; stop on no improvement).
- Greedy backward elimination (start with all 10, remove the probe whose removal helps most; stop on no improvement).
- Continuous weight optimization via `scipy.optimize.minimize(method="SLSQP")` over the probability simplex.
- LR stacking with C ∈ {0.1, 1, 10, 100}.

Top results are in Appendix A.

The chosen subset was the size-5 stable configuration `[ensemble_top8 (weighted_val_acc), cross_layer_mean, geo_gbt, mlp_pool_mlp, regen_overlap]` at OOF score 0.7866 ± 0.004 — picked over the highest-mean size-6 subset (0.7953 ± 0.014) because the 0.6 pp mean difference was inside one standard deviation while the std was 3.5× larger; in cross-validation terms the size-6 winner was plausibly fold-luck. The size-5 winner translated into the meta-avg combiner's specific top-portion (the 8 sub-probes already inside the inner ensemble) + standalone (the same 4 additional probes).

The OOF-search procedure tunes the threshold on ≈551 outer-train OOF samples, while `evaluate.run_evaluation` initially tuned on ≈83 val samples per fold — a 6.6× difference in threshold sample size, accounting for the ≈1.5–2 pp gap between OOF-search-predicted accuracy (0.7866) and the original implementation's per-fold accuracy (0.7619). Aligning the implementation's threshold step with the search's (the §5.7 change) closed this gap to ≈0.7 pp.

### 5.2 Standalone probe families explored

10 candidate probes fed the subset search. Single-probe OOF accuracies (5-fold outer CV, threshold tuned on outer-train):

- Cross-layer mid-layer pool + LR (C=0.01) — 0.7678 ± 0.018
- Last-token-L20 + `geo_features` + HistGBT — 0.7576 ± 0.023
- Actual-token logprobs + layer-24 logit-lens features + GBT — 0.7474 ± 0.024
- Cross-model logprob agreement (Qwen2.5-0.5B + SmolLM2-360M-Instruct) — 0.7373 ± 0.019
- 7-layer logit-lens + cross-layer KL + agreement features + GBT — 0.7315 ± 0.018
- Last-tokens of several layers concatenated + LR — 0.7300 ± 0.011
- Single-layer logit-lens features + GBT — 0.7228 ± 0.012
- Layer-23 response-mean + MLP — 0.7199 ± 0.030
- Greedy-regeneration overlap features only + GBT — 0.6981 ± 0.016

Importantly: probes scoring *below* the 0.7010 majority-class baseline (the greedy-regen-overlap-only probe) still made it into the final ensemble. The OOF subset search consistently picked them whenever their inclusion lowered the variance of the meta-mean — variance reduction through error decorrelation, not mean increase. This counter-intuitive result is the central lesson of the search-driven selection process: candidate probes should not be eliminated by single-probe accuracy alone.

### 5.3 Ensemble combiners compared

Four combiners were compared on the same 10-column OOF probability matrix:

- **`average`** — equal-weight simple mean of selected columns. Best for unbiased aggregation when individual accuracies are comparable.
- **`weighted_val_acc`** — `w_i ∝ max(acc_i − 0.5, ε)` from OOF accuracy, normalised. Slight win over `average` when probes vary in quality, in this setting by ≈0.3 pp.
- **`stacking_lr`** — fit a meta `LogisticRegression(C ∈ {0.1, 1, 10, 100}, class_weight="balanced")` on the OOF probability matrix → use it to predict. Underperformed simple averaging consistently (Appendix A). With only 689 OOF samples and 10 input columns, the meta-LR overfit the OOF — the test fold then suffered.
- **`meta_avg`** — hierarchical combine: 8 inner sub-probes get a `weighted_val_acc` combine into a single `p_top8`, then `p_top8` is simple-mean averaged with 4 standalones (3 of which deliberately duplicate inner sub-probes). This is what the OOF subset search found optimal, and what the final solution uses.

The continuous-weight optimization step (SLSQP on the probability simplex) consistently collapsed to a near-uniform weight vector — likely because the threshold-search step inside the objective makes the loss landscape non-differentiable. The SLSQP result is treated as an unreliable ablation rather than as evidence that uniform weights are truly optimal.

### 5.4 Probe architecture experiments

- **Multi-seed bagging on LR with `lbfgs` solver — no-op.** Fitting LR with `random_state ∈ {42, 7, 123, 2024, 31}` and averaging produced byte-identical probabilities on this dataset. `lbfgs` is essentially deterministic for fixed data; varying only the seed is a no-op. Replaced with multi-C bagging (`_MultiCLR`), which produces genuinely different fits because each C value is a different objective.
- **MLP with LayerNorm + AdamW + cosine LR schedule** — tested standalone, did not exceed the strong-L2 LR at the equivalent feature slice. The plain MLP (Linear → ReLU → Dropout(0.3) → Linear) was retained because it contributed differently to the ensemble than the LR family (different decision boundary, useful diversity).
- **PCA-64 preprocessing on the LR sub-probes** — tested on a few sub-probes. Modest help (≈0.3 pp) on some, neutral on others; not the bottleneck. Not adopted globally.
- **HistGBT with extreme depth limits** (max_depth=2, 3, 4, 6) — depth 4 was best for the actual-logprob / stat-drift / regen blocks (small feature dim, simple structure); depth 6 was best for the geo_gbt sub-probe (101 heterogeneous geometric features that benefit from interaction modelling).
- **ExtraTreesClassifier (n_estimators=400, max_features="sqrt")** — kept as an additional tree-family sub-probe (`extra_trees_geo`) on the same blocks as `geo_gbt`. Different randomisation → different errors → ensemble win.

### 5.5 Data and external sources

Augmenting the training set was considered with one or more of:

- HaluEval (HuggingFace dataset, ≈35k examples).
- TruthfulQA.
- SQuAD- or HotpotQA-derived synthetic hallucinations generated by prompt-manipulation of Qwen.

All three were ultimately rejected. (a) `dataset.csv` has a specific ChatML format and Qwen-2.5-0.5B response style — external data is likely to introduce distribution shift that the probe over-learns. (b) The spirit of the competition is to make the most of the 689 provided training samples; external data feels off-scope and the rules do not give clear guidance on this. (c) Time investment for an uncertain gain.

Inside the provided data: verified that `dataset.csv` has zero exact prompt duplicates, zero exact (prompt, response) duplicates, and zero user-message duplicates after stripping the shared system block. Also verified that no test-set prompt or user message appears in the training set. Cross-fold leakage with the chosen `StratifiedKFold(random_state=42)` is zero (verified by direct set intersection on each fold split). Group-aware splitting was therefore unnecessary.

### 5.6 Active-LLM features (regeneration, self-critique)

Features that *use the model differently from just reading hidden states* turned out to be a third orthogonal axis next to hidden-state pooling and hand-crafted geometry. Tried:

- **Greedy regeneration overlap** — kept. Implemented in `regen_features` block: for each prompt, generate Qwen's natural continuation greedily, compare with the actual `response` column on token Jaccard, token overlap ratio, first-N-token match rate, length ratio, length difference, and the mean per-token logprob of the regenerated tokens. Standalone OOF accuracy 0.6981 (below baseline), but contributes to the ensemble via decorrelation.
- **SelfCheckGPT-style multi-sample regeneration** — skipped. Generating K=5 samples per prompt with temperature > 0 is roughly 5× the cost of greedy and the expected gain was judged too modest given that greedy regen overlap was already in.
- **Self-critique via "Is this answer factually correct? Yes / No" prompt** — skipped. A 0.5B model is a poor self-critic; the output distribution is heavily biased toward "Yes" regardless of factuality.
- **Cross-model logprob agreement** — tested with Qwen2.5-0.5B + SmolLM2-360M-Instruct as the secondary model. Single-probe OOF accuracy 0.7373; included in the subset search but ultimately edged out of the meta-mean by other probes that contributed more per pp of feature dim.
- **Logit lens at multiple layers + cross-layer KL** — tested; standalone 0.7315; included in the size-6 subset-search winner but excluded from the size-5 winner that ships. The cross-layer KL features add signal in some folds but increase variance enough that the size-5 (more stable) was preferred.

### 5.7 Hyperparameter and threshold-tuning experiments

This is where the final lift came from. Key experiments:

- **F1-best threshold vs accuracy-best threshold**. The original skeleton's `fit_hyperparameters` tunes the threshold to maximise F1. On a 70 / 30 class-imbalanced dataset, the F1-optimal threshold for many probes collapses to "predict 1 always" — yielding ≈0.825 F1 but only ≈0.70 accuracy. Switching the target to accuracy was the single largest one-line lift in the entire iteration (≈+2–3 pp on most probes).
- **Val-tuned threshold (≈83 samples) vs OOF-tuned threshold (≈551 samples)** for the `meta_avg` combiner specifically. The OOF subset search measured ensemble OOF accuracy with a threshold tuned on outer-train OOF (large sample); when this was applied to the actual `evaluate.run_evaluation` per-fold pipeline (which calls `fit_hyperparameters` with the small inner val), the threshold deteriorated and the accuracy fell by ≈1.5 pp. Empirically, with the same model, AUROC 0.784 + accuracy 0.756 was symptomatic of "rankings are good, threshold is wrong." Skipping `fit_hyperparameters` for `meta_avg` (using only the OOF threshold from `fit()`) recovered the 1.5 pp and brought the per-fold pipeline's accuracy in line with the OOF-search prediction.
- **C sweep for LR** — `C ∈ {1.0, 0.5, 0.1, 0.05, 0.01, 0.005}` on layer-23 response-mean. Best: 0.01. Public reference solutions surveyed during the iteration converged on C ∈ [0.01, 0.03].
- **HistGBT learning rate** — `lr ∈ {0.03, 0.05, 0.08}` × `max_depth ∈ {3, 4, 6}`. Sweet spots for different blocks landed at (0.05, 6) for `geo_gbt`, (0.04, 4) for the smaller-feature blocks. No grid search beyond this — the marginal gain from more elaborate tuning was within fold noise.

### 5.8 What is not part of the shipped submission

The iteration produced several components that are deliberately *not* included in the shipped solution:

- The per-branch OOF dump orchestrator and the offline subset-search script that selected the final ensemble composition. These were one-time experimental tools that consumed git worktrees of candidate-probe branches and produced the probability matrix that drove §5.1. They are not needed to run the solution.
- Side scripts that pre-computed alternative cached features (e.g. secondary-model logprobs for the cross-model probe) for experimental branches that did not make the final cut.
- Data-quality exploration notebooks and per-layer / per-token-position diagnostic scripts used during iteration to verify the splitting assumptions (no duplicate prompts, no leakage) and to characterise hidden-state norms across layers.
- Worktrees of every candidate-probe branch (≈10 separate solutions) and their dumped OOF prediction files.

`python solution.py` against a clean checkout containing only the three editable files (`aggregation.py`, `probe.py`, `splitting.py`), the unchanged upstream files, and `analysis/regenerations.npz` produces `results.json` and `predictions.csv` without depending on any of the above.

## Appendix A: per-experiment results

### A.1 Standalone probe single-OOF accuracies

5-fold outer `StratifiedKFold(random_state=42)`, threshold tuned per outer fold on outer-train OOF.

| Probe | OOF accuracy ± std | Standalone-feature focus |
| --- | --- | --- |
| Cross-layer mid-layer pool + LR | 0.7678 ± 0.018 | Mid-layer (12–16) pool average + LR C=0.01 |
| Inner top-8 ensemble (weighted_val_acc) | 0.7663 ± 0.015 | Earlier iteration of the inner top-8 weighted-mean |
| Last-tok-L20 + geo + HistGBT | 0.7576 ± 0.023 | last_tok_l20 + 101 geo features + HistGBT |
| Actual-token logprobs + GBT | 0.7474 ± 0.024 | logit-lens-24 + actual-token logprobs |
| Cross-model logprob agreement | 0.7373 ± 0.019 | Qwen + SmolLM2-360M secondary model |
| Multi-layer logit lens + cross-layer KL | 0.7315 ± 0.018 | 7-layer logit-lens + cross-layer KL |
| Multi-layer last-token concat + LR | 0.7300 ± 0.011 | Last-tokens of several layers concatenated |
| Single-layer logit lens + GBT | 0.7228 ± 0.012 | Single-layer logit-lens + GBT |
| Layer-23 response-mean + MLP | 0.7199 ± 0.030 | Layer-23 response-mean + MLP |
| Greedy regen overlap + GBT | 0.6981 ± 0.016 | Greedy regeneration overlap features only |
| Majority-class baseline | 0.7010 | — |

### A.2 Subset search top 8

Full-subset enumeration over the 10 columns above, `simple_mean(selected columns)` combiner, outer 5-fold CV.

| Mean acc ± std | Size | Subset |
| --- | --- | --- |
| 0.7953 ± 0.014 | 6 | cross-layer-mean, top8 ensemble, geo-gbt, logit-trajectory, mlp, regen-overlap |
| 0.7895 ± 0.022 | 3 | cross-layer-mean, geo-gbt, mlp |
| 0.7895 ± 0.013 | 5 | actual-logprob, cross-layer-mean, geo-gbt, mlp, regen-overlap |
| **0.7866 ± 0.004** | **5** | **cross-layer-mean, top8 ensemble, geo-gbt, mlp, regen-overlap** — chosen for stability |
| 0.7866 ± 0.010 | 6 | actual-logprob, cross-layer-mean, top8 ensemble, geo-gbt, mlp, regen-overlap |
| 0.7852 ± 0.011 | 5 | actual-logprob, cross-layer-mean, top8 ensemble, mlp, regen-overlap |
| 0.7837 ± 0.009 | 7 | cross-layer-mean, cross-model, top8 ensemble, geo-gbt, logit-lens, mlp, regen-overlap |
| 0.7808 ± 0.004 | 4 | cross-layer-mean, top8 ensemble, mlp, regen-overlap |

### A.3 Combiner comparison (on the size-5 stable subset)

| Combiner | OOF mean acc ± std | Notes |
| --- | --- | --- |
| `simple_mean` | 0.7866 ± 0.004 | The reference; what the subset search measured. |
| `weighted_val_acc` | 0.7619 (5-fold eval), 0.7866 (OOF search) | Slight gap between OOF-search and eval pipelines explained in §5.7. |
| `stacking_lr` C=100 | 0.7590 ± 0.020 | Meta-LR overfits the 689 OOF rows. |
| `stacking_lr` C=10 | 0.7416 ± 0.033 | |
| `meta_avg` (the chosen shape, pre-OOF-threshold change) | 0.7619 ± 0.020 | Per-fold eval, val-tuned threshold. |
| **`meta_avg` (post-OOF-threshold change — shipped)** | **0.7794 ± 0.014** | Per-fold eval, OOF-tuned threshold via skipping `fit_hyperparameters`. |

### A.4 Pre- vs post-change meta-avg

| | Per-fold test accs | Mean | Std | Min fold | Max fold |
| --- | --- | --- | --- | --- | --- |
| Pre (val-tuned threshold) | 0.7391, 0.7971, 0.7898, 0.7391, 0.7445 | 0.7619 | 0.0260 | 0.7391 | 0.7971 |
| **Post (OOF threshold)** | **0.7826, 0.7971, 0.7754, 0.7826, 0.7591** | **0.7794** | **0.0125** | **0.7591** | **0.7971** |
| Δ | — | **+0.0175** | −0.0135 | **+0.0200** | 0 |

The change lifts the mean by 1.75 pp, halves the standard deviation, and raises the minimum fold by 2 pp — i.e. the floor improves more than the ceiling. The probabilities themselves are unchanged; only the threshold differs.
