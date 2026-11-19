# Evaluation and submission notes

## Live deployment snapshot

Checked on **2026-11-19** at `http://15.206.67.161/`. The public `/api/v1/model-info` endpoint responded successfully and reported:

| Item | Reported value |
|---|---:|
| Configured checkpoint | `multicountry_india_custom_18class.pth` |
| Crop classes | 18 |
| Training parcels | 11,149 |
| Held-out test parcels | 2,230 (20% split) |
| Single-draw accuracy | 66.82% |
| 16-draw ensembled accuracy | 72.11% |
| Macro F1 | 0.53 |
| Weighted F1 | 0.71 |

The reported 16-draw ensemble matches the deployed prediction preprocessing. This is a random held-out split from the collected parcels, not an independent test across unseen countries. Metrics are reported by the running service; the underlying training parcels and predictions are not included in this package, so this report is not an independent recomputation.

The API reported these per-class F1 scores:

| Crop class | F1 | Test examples |
|---|---:|---:|
| Rapeseed | 0.95 | 210 |
| Maize | 0.91 | 240 |
| Potatoes | 0.81 | 215 |
| Barley | 0.78 | 210 |
| Sunflower | 0.78 | 131 |
| Vineyards | 0.75 | 104 |
| Wheat | 0.70 | 252 |
| Meadow | 0.66 | 210 |
| Triticale | 0.63 | 210 |
| Fruit | 0.62 | 210 |
| Sugarcane | 0.52 | 27 |
| Fallow | 0.49 | 30 |
| Rice | 0.38 | 14 |
| Nuts | 0.35 | 96 |
| Mustard | 0.20 | 36 |
| Lentil | 0.00 | 15 |
| Gram | 0.00 | 11 |
| Garlic | 0.00 | 9 |

At the same check, `/api/v1/parcels/regions` and `/api/v1/demo/parcels` responded successfully; the demo endpoint returned a 26-feature GeoJSON collection. These checks verify the public pages and read-only API endpoints, not a fresh paid/live Sentinel Hub prediction.

## Training curves and notebooks

**The epoch-by-epoch training/validation history for the deployed 18-class checkpoint is not present in the submitted files.** The current local training script prints loss and validation accuracy while it runs, but does not save that history. Its `--dump-json` option saves final evaluation metrics only. The multi-country notebooks have no saved cell outputs. Consequently, a training/validation curve for this checkpoint cannot be honestly plotted from the available artifacts.

`backend/training/crop_classifier_training.ipynb` contains outputs from earlier experiments, but those are not the training history for the deployed 18-class model. They should not be presented as that model's curve. The saved `.pth` files are model weights, not logs or datasets.

To produce an authentic curve in a future retraining, preserve the per-epoch training loss and validation accuracy from `backend/training/eurocrops_pipeline/run_train.py` when training is run with its fetched training data. The fetched parcel checkpoints, private hand-labeled inputs, and prior console logs are not included here; a full retrain is therefore not reproducible from this ZIP alone.

## Local report versus deployed report

`backend/app/data/model_stats.py` contains a checked-in, earlier evaluation snapshot (8,351 parcels; 1,671 test examples). The running deployment returned the newer figures above (11,149 parcels; 2,230 test examples). They describe different evaluation runs and must not be combined or treated as identical. The checked-in report is retained as part of the source; use the dated deployment snapshot for the live demonstration.

The deployment reports Denmark and Brittany in its training data; the checked-in local snapshot does not. The ZIP includes the configured 18-class checkpoint and the complete project source that is present in this workspace, but the available files do not establish that the local checkpoint is bit-for-bit the same artifact currently loaded by the deployment.

## Files and limitations

The project ZIP includes the frontend, backend, training code/notebooks, dependency manifests, documentation, and the saved checkpoints present in the workspace. It intentionally excludes credentials, downloaded training data, local user-provided parcel data, virtual environments, and generated build output. Credentials must be supplied by the evaluator in `backend/.env`; see [README.md](./README.md).

