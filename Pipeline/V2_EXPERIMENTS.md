# SS-VLM v2 Experiments

These files are additive. They do not replace `Pipeline/SS-VLM_Pipeline.py` and they write to `outputs_v2/` so the existing `outputs/` results remain comparable.

## New pipeline

`Pipeline/SS-VLM_Pipeline_v2.py` adds:

- `--model_variant plain_vit`: true plain ViT baseline using the `[CLS]` token.
- `--model_variant vit_gem`: stronger ViT+GeM baseline matching the old pooling style but without SFRA.
- `--model_variant sfra_v2`: corrected SFRA initialization with a learnable residual scale.
- `--seed`: reproducible training/data-shuffle seed.
- `--backbone_lr` and `--head_lr`: differential learning rates.
- `--lambda_cont` and `--supcon_start_epoch`: SupCon sweep and warmup.
- `--au_fusion_beta`: optional OpenFace AU-score fusion during RAG prediction.
- `--au_model_path`: optional learned AU-to-emotion calibrator. When provided, RAG uses learned AU probabilities instead of the manual FACS-prior fallback.

## Learned AU evidence

The final AU design has two complementary pieces:

1. `tools/train_au_emotion_tabular.py` trains the prediction-time AU prior used in RAG fusion.
2. `tools/learn_au_emotion_mapping.py` trains a sparse interpretable model to summarize which AUs support each emotion.

The prediction-time AU prior is the one passed to `Pipeline/SS-VLM_Pipeline_v2.py` with `--au_model_path`. The sparse mapping table is used for paper explanation, AU support analysis, and report grounding.

Train the learned AU prior:

```bash
python tools/train_au_emotion_tabular.py \
  --au-csv outputs/rafdb_openface2_aus.csv \
  --output-dir outputs_v2/au_calibrator \
  --run-name au_both_rf_seed42 \
  --feature-set both \
  --model random_forest \
  --selection-metric accuracy \
  --n-estimators 1200 \
  --seed 42
```

Learn the interpretable AU-emotion support table:

```bash
python tools/learn_au_emotion_mapping.py \
  --au-csv outputs/rafdb_openface2_aus.csv \
  --output-dir outputs_v2/au_mapping \
  --run-name learned_au_mapping_intensity_elasticnet \
  --feature-set intensity \
  --seeds 42,123,2026,7,99 \
  --permutation-repeats 10 \
  --top-k 6
```

## Recommended order

Run the compact sweep first:

```bash
sbatch train_ssvlm_v2_sweep.job
```

After it finishes, evaluate the full paper set with learned-AU RAG fusion:

```bash
sbatch rag_fusion_ssvlm_v2_learned_au_all_seeds.job
```

The older manual-AU fallback RAG job is still available for comparison:

```bash
sbatch rag_fusion_ssvlm_v2_all_seeds.job
```

The all-seeds RAG job evaluates:

```text
plain_vit: seeds 42, 123, 2026
vit_gem: seeds 42, 123, 2026
sfra_v2: seeds 42, 123, 2026
```

If the compact sweep shows SFRA is competitive, run the seed sweep:

```bash
sbatch ablation_ssvlm_v2_seed_sweep.job
```

## Output layout

```text
outputs_v2/
  models/
    sweep/
    seed_sweep/
  metrics/
    sweep/
    rag_fusion/
    rag_fusion_learned_au/
    seed_sweep/
  au_calibrator/
    au_both_rf_seed42/
  au_mapping/
    learned_au_mapping_intensity_elasticnet/
```
