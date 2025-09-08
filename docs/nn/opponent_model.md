## Opponent Modeling Network

Implements a CNN (see `opponent_model.py`) that predicts opponent ship placement heatmaps given multi-channel board state including engineered planes for opponent tendency and board edges.

### Architecture
- Input: (10,10,5) planes = [miss, hit, unknown, opponent_tendency, edge].
- Body: 4× Conv2D blocks with 64→128→64 channels + BatchNorm.
- Output: Conv2D(1,1×1,sigmoid) ⇒ ship heatmap.
- Loss: binary cross-entropy; Optimizer: Adam(1e-3).

### Data
- Synthetic generator yields diverse opponents (edge/corner/center/diagonal/uniform) with noise; real data loader can consume `data/opponent_data.pkl` when available.
- Train/val split 80/20; callbacks: EarlyStopping, ModelCheckpoint, TensorBoard. Saved at `models/opponent_model.h5` with history in `models/opponent_model_history.pkl`.

### Usage
- Train: `python opponent_model.py` (prompts for retrain). Visualize predictions saved to `logs/opponent_model_predictions.png`.
- Profiles persisted at `models/opponent_profiles.pkl` store behavioural frequencies and are updated post-game via helper APIs.

### Integration points
- Profiles can be folded into density prior or as an extra prior plane in agent blending.
- AIAgent3’s opponent profiling updates orientation and attack pattern counts during gameplay; persisted for future sessions.

### Research notes
- Synthetic-to-real gap can be closed by bootstrapping from synthetic profiles then fine-tuning on logged human/AI opponents.
- Additional channels (e.g., shoreline/cluster priors) and curriculum sampling can improve performance.
