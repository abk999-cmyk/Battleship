## Reproducibility Guide

This guide ensures consistent setup, execution, and reporting.

### Environment setup (venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- macOS/Apple Silicon: optionally run `python scripts/setup_tensorflow.py` for TF metal acceleration.

### Deterministic seeds
```python
import random, numpy as np
random.seed(1337)
np.random.seed(1337)
```
- For scripts using multiprocessing (GA, RL), seed in worker initializers where required.

### Model training
```bash
python training/generate_dataset.py --games 50000 --workers 8 --out data/battleship_supervised.pkl
python training/train_heatmap.py
python training/rl_finetune.py --games 20000 --workers $(sysctl -n hw.ncpu) --batch 256
```

### GA evolution
```bash
python training/ga_optimizer.py --pop 40 --gens 50 --cpus 8
```

### Benchmarking
```bash
python apps/main.py --games 1000
# or
python apps/main.py --watch --opp ultimate --delay 0.3
```

### Dashboard
```bash
python apps/battleship_dashboard.py
```

### Data locations
- Models: `models/`
- Datasets: `data/`
- Logs and plots: `logs/`

### Notes
- Ensure TensorFlow 2.17.0 per `requirements.txt`.
- Run heavy jobs under venv; pin CPU counts on laptops to avoid thermal throttling.
