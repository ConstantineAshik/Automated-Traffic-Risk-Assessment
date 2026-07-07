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

The three supplied YOLO checkpoints load successfully and are treated as
COCO-trained object detectors. The pipeline therefore keeps all COCO labels
instead of pretending the checkpoints know Dhaka-only classes such as rickshaw
or CNG. Risk is calculated from forward-path objects, tracking motion, optical
flow, Time-to-Collision (TTC), and traffic-jam context.

Close distance by itself is not considered dangerous. A nearby vehicle in slow,
stable congestion is treated differently from a nearby vehicle that is rapidly
expanding in the rider's forward path.

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

By default the detector runs in COCO mode:

```python
PipelineConfig(detection_dataset="coco", detect_all_coco_objects=True)
```

This lets the model report any COCO object it sees. Only objects inside the
middle-lower rider path area strongly affect following-distance/TTC risk;
objects beside the rider are kept in the detection log with lower risk weight.

## Test

```powershell
python -m pytest -q
```

## Train a real risk classifier

Each CLI analysis creates `<video-name>_predictions.csv` with detector-derived
features and a blank `human_label` column. Label samples as `SAFE`, `CAUTION`,
or `DANGER`, preserve a unique `ride_id` for every independent ride, then first
measure the rule-based baseline:

```powershell
python -m training.evaluate_risk_model labeled_ride_*.csv
```

Blank `human_label` rows are ignored, so you can label the most useful samples
first instead of completing every frame in one sitting. After you have at least
three independent rides with all three classes represented, train:

```powershell
python -m training.train_risk_model labeled_ride_*.csv `
  --output models\risk_model.joblib
```

The CLI and dashboard automatically load `models/risk_model.joblib` after the
training command succeeds. Re-evaluate the trained classifier directly with:

```powershell
python -m training.evaluate_risk_model labeled_ride_*.csv `
  --model models\risk_model.joblib `
  --output models\risk_model.eval.json
```

For custom model locations, enable it explicitly:

```python
PipelineConfig(structured_risk_model_path="models/risk_model.joblib")
```

See [docs/DATASET_AND_EVALUATION.md](docs/DATASET_AND_EVALUATION.md) before
collecting or splitting data.
