# Battleship AI Research Platform Documentation

This documentation covers the entire system in depth: architecture, agents, training, evaluation, and reproducibility.

- Overview and System Architecture → `overview.md`
- Game Environment and Mechanics → `environment.md`
- Agents
  - Agent 2 Architecture and Policy → `agents/agent2.md`
  - Agent 3 Architecture, Training, and Move Selection → `agents/agent3.md`
  - Agent 4 (GA-Optimized Agent 3) → `agents/agent4.md`
- Neural Networks
  - Supervised Heatmap CNN → `nn/heatmap_model.md`
  - Opponent Modeling Network → `nn/opponent_model.md`
  - Meta-Learner → `nn/meta_learner.md`
- Training Pipelines → `training/pipelines.md`
- Evaluation Protocols and Metrics → `evaluation/metrics.md`
- Dashboard and Webapp → `ui/dashboard_webapp.md`
- API and Module Reference → `api/reference.md`
- Reproducibility (env, seeds, commands) → `repro/guide.md`
- Experiment Reports and Results → `reports/experiments.md`
- Change Log and Release Notes → `CHANGELOG.md`

Quick links:
- Agents: [Agent 2](agents/agent2.md) · [Agent 3](agents/agent3.md) · [Agent 4](agents/agent4.md)
- Training: [Pipelines](training/pipelines.md) · [Heatmap CNN](nn/heatmap_model.md) · [RL Fine-tune](../rl_finetune.py)
- Evaluation: [Metrics](evaluation/metrics.md) · [Dashboard](ui/dashboard_webapp.md)
