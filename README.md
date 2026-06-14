# SS-VLM

SS-VLM: A Spectral-Symbolic Vision-Language Framework for Auditable Facial Expression Recognition.

## Repository Role

This repository is the **latest updated implementation** of the SS-VLM project:

- Latest updated repository: https://github.com/rijju-das/SS-VLM
- Base/original repository: https://github.com/Moonquakeyu/Spectral-Symbolic-VLM

The base repository preserves the original development code and exploratory material. This repository contains the manuscript-aligned implementation, final v2 experiment scripts, learned-AU RAG fusion pipeline, audit/evaluation utilities, and the result artifacts used for the current paper.

## Project Story

SS-VLM reframes facial expression recognition from a black-box label prediction task into an auditable affective reasoning pipeline. The final implementation combines:

- **SFRA visual encoding** for localized high-frequency facial refinement.
- **Dual-head training** for classification and retrieval-friendly embeddings.
- **Prototype retrieval** over RAF-DB train-set centroids.
- **Learned AU calibration** from OpenFace 2.0 AU intensity/presence features.
- **Weighted RAG fusion** of classifier, retrieval, and AU-prior probabilities.
- **Constrained report generation** with evidence checks to reduce unsupported AU or clinical claims.

The latest manuscript results are based on the learned-AU v2 pipeline, not the older base-only pipeline.

## Key Code

- `Pipeline/SS-VLM_Pipeline_v2.py`: final v2 training, retrieval, learned-AU RAG fusion, and reporting pipeline.
- `Pipeline/V2_EXPERIMENTS.md`: v2 experiment notes and command overview.
- `tools/extract_openface2_aus.py`: RAF-DB OpenFace 2.0 AU extraction.
- `tools/train_au_emotion_tabular.py`: learned RandomForest AU prior.
- `tools/learn_au_emotion_mapping.py`: sparse AU-emotion mapping for interpretation.
- `tools/analyze_v2_reliability.py`: per-class, correction-flow, and ECE summaries.
- `tools/evaluate_hallucination_metrics.py`: report-audit and faithfulness metrics.
- `tools/plot_*`: figure-generation utilities for confusion matrices, reliability, k-sensitivity, AU figures, Grad-CAM, and zero-shot VLM results.

## Manuscript-Aligned Results

The final result artifacts are organized as follows:

- `outputs/rafdb_openface2_aus.csv`: OpenFace 2.0 AU features used by learned AU calibration.
- `outputs_v2/models/seed_sweep/`: final seed-sweep checkpoints for Plain ViT, ViT+GeM, and SFRA.
- `outputs_v2/metrics/seed_sweep/`: classifier training histories and summaries.
- `outputs_v2/metrics/rag_fusion_learned_au/beta005/`: final learned-AU RAG predictions, reports, summaries, and k-sensitivity files.
- `outputs_v2/metrics/reliability_analysis_learned_au/`: manuscript-aligned correction-flow, per-class, McNemar-adjacent, and ECE summaries.
- `outputs_v2/au_calibrator/au_both_rf_seed42/`: selected RandomForest AU prior artifacts.
- `outputs_v2/au_mapping/learned_au_mapping_intensity_elasticnet/`: sparse learned AU-emotion interpretation artifacts.
- `outputs_v2/figures/`: generated result figures.
- `outputs_vlm/zero_shot/`: zero-shot InstructBLIP/LLaVA RAF-DB outputs.
- `Figure/`: final paper-ready figure PDFs copied from the manuscript directory.

## Reproduction Outline

1. Create the SS-VLM environment using `environment-ssvlm.yml` or `requirements.txt`.
2. Use `outputs/rafdb_openface2_aus.csv` directly, or regenerate it with `tools/extract_openface2_aus.py`.
3. Run v2 classifier/seed sweeps with `train_ssvlm_v2_sweep.job` and `baseline_v2_seed_sweep.job`.
4. Train or inspect AU models using `tools/train_au_emotion_tabular.py` and `tools/learn_au_emotion_mapping.py`.
5. Run final learned-AU RAG evaluation with `rag_fusion_ssvlm_v2_learned_au_all_seeds.job`.
6. Regenerate reliability, correction-flow, report-audit, and figure outputs with the scripts in `tools/`.

## Final Manuscript Framing

The paper should cite both repositories:

- `Spectral-Symbolic-VLM` as the base/original project repository.
- `SS-VLM` as the latest updated manuscript-aligned implementation.

This distinction keeps the development lineage transparent while making the final reproducible implementation easy to locate.
