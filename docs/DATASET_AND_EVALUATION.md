# Dataset and evaluation protocol

## 1. Establish the label definition first

Use contiguous 2–5 second events, not isolated images. Assign one risk label:

- `SAFE`: normal riding with adequate response margin
- `CAUTION`: a hazard is present, but no immediate evasive response is needed
- `DANGER`: collision risk, distraction, or road conflict requires immediate action

Also record factor labels such as phone distraction, cut-in, pedestrian
conflict, short following distance, rain, glare, and congestion. Have two
reviewers independently label an initial overlap set and resolve disagreements.

## 2. Collect representative footage

Include different riders, cameras, roads, times of day, weather, congestion,
and video resolutions. Deliberately retain hard negatives:

- Close but stationary traffic jams
- Parked buses and trucks
- Pedestrians beside the road who are not crossing
- Mounted phones
- Bright dry roads that resemble wet-road glare

Start with 1,000–3,000 labeled clips from dozens of independent rides. This is
a starting target, not a guarantee of adequate coverage.

## 3. Prevent leakage

Split by complete `ride_id`. Frames or clips from one ride must never appear in
both training and testing. Prefer holding out riders, roads, and dates as well.
The included training command uses grouped holdout splitting for this reason.

Do not tune thresholds on the final test rides.

## 4. Train the risk baseline

Run the CLI once per ride to produce a feature CSV. Give every file a stable,
unique `ride_id`, fill its `human_label` column, and train with:

```powershell
python -m training.train_risk_model ride_001.csv ride_002.csv ride_003.csv
```

The command writes both a Joblib model and a JSON metrics file. The bundled mock
TF-IDF model is disabled by default and should not be used for accuracy claims.

## 5. Evaluate what matters

Report at least:

- Danger recall and danger precision
- Macro F1 and balanced accuracy
- Confusion matrix
- False danger alerts per minute
- Event-level detection rate, not only frame accuracy
- Results by day/night, weather, congestion, camera, and road type
- Confidence intervals across independent rides

For deployment threshold selection, prioritize danger recall while setting an
acceptable false-alert rate. Review every false negative.

## 6. Fine-tune object detection properly

The current checkpoints expose COCO labels. Create a Dhaka object-detection
dataset with bounding boxes for the classes the risk logic actually needs,
including rickshaw/CNG if required. Keep train, validation, and test images
grouped by source video.

Record for every checkpoint:

- Dataset version and class map
- Train/validation/test ride IDs
- Training command and hyperparameters
- Per-class precision, recall, and mAP
- Performance on the locked Dhaka test set

Only call a checkpoint “Dhaka-trained” when this provenance accompanies it.
