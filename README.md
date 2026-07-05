# Automated Traffic Risk Assessment

Computer-vision analysis of motorcycle riding behavior in Dhaka traffic.

The pipeline now combines:

- An ensemble of `models/yolo11n.pt`, `models/yolo11m.pt`, and `models/yolo12m.pt`
- Confidence-aware same-class box fusion
- Temporal object identity, time-to-collision, and lateral-motion estimation
- Resolution- and sample-rate-normalized motion features
- Interpretable risk scoring and non-amplifying temporal smoothing
- Optional structured risk classification trained from human labels

## Important model limitation

The three supplied YOLO checkpoints load successfully, but expose the standard
80 COCO labels. They do not expose Dhaka-specific labels such as rickshaw or
CNG, and two identify COCO dataset metadata. They are used as a general object
detection ensemble. Do not describe them as Dhaka-fine-tuned until their
training/evaluation records demonstrate that.

Inspect checkpoint metadata at any time:

```powershell
python tools\inspect_models.py
```

## Install and run

```powershell
python -m pip install -r requirements.txt
python main.py <path-to-video>
```

Or launch the dashboard:

```powershell
streamlit run app.py
```

The default is the accuracy-oriented three-model ensemble. On a CPU it can be
slow. A fast profile can be created in code with:

```python
PipelineConfig(model_paths=("models/yolo11n.pt",))
```

For a compromise, use the nano and one medium checkpoint:

```python
PipelineConfig(
    model_paths=("models/yolo11n.pt", "models/yolo12m.pt"),
    ensemble_min_model_votes=1,
)
```

## Test

```powershell
python -m pytest -q
```

## Train a real risk classifier

Each CLI analysis creates `<video-name>_predictions.csv` with detector-derived
features and a blank `human_label` column. Label samples as `SAFE`, `CAUTION`,
or `DANGER`, preserve a unique `ride_id` for every independent ride, then train:

```powershell
python -m training.train_risk_model labeled_ride_*.csv `
  --output models\risk_model.joblib
```

The CLI and dashboard automatically load `models/risk_model.joblib` after the
training command succeeds. For custom model locations, enable it explicitly:

```python
PipelineConfig(structured_risk_model_path="models/risk_model.joblib")
```

See [docs/DATASET_AND_EVALUATION.md](docs/DATASET_AND_EVALUATION.md) before
collecting or splitting data.
