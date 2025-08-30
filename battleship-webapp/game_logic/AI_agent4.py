"""
AI_agent4.py – GA‑aware clone of AI_agent3
=========================================

Drop this file into the same package as `AI_agent3.py`.
`AIAgent4` inherits *all* behaviour from `AIAgent3` but automatically
loads the GA‑optimised meta‑weights stored in `models/ga_weights.json`.

Usage
-----
```python
from AI_agent4 import AIAgent4 as AIAgent  # <- switch import like this

game = BattleshipGame(AIAgent(), Opponent())
result = game.play()
```

As soon as you run `ga_optimizer.py`, it writes a weight file in the
`models/` folder.  Every subsequent instantiation of `AIAgent4` will
pick those values up at start‑up, so **all simulations use the evolved
blend automatically**.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from .AI_agent3 import AIAgent3

logger = logging.getLogger(__name__)


class AIAgent4(AIAgent3):
    """AIAgent3 + automatic GA weight loading."""

    GA_PATH = os.path.join("models", "ga_weights.json")

    def __init__(self, name: str = "UltimateBattleshipAgent‑GA", *args: Any, **kwargs: Any):
        super().__init__(name, *args, **kwargs)
        self._load_ga_weights()

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _load_ga_weights(self) -> None:
        """If GA weight file exists, merge it into self.meta_weights."""
        if not os.path.exists(self.GA_PATH):
            logger.info("[GA] No GA weight file found – using default meta‑weights.")
            return

        try:
            with open(self.GA_PATH, "r", encoding="utf‑8") as f:
                ga_weights = json.load(f)
            # Only update keys that already exist in meta_weights
            valid_ga_weights = {
                k: float(v) for k, v in ga_weights.items() if k in self.meta_weights
            }
            if valid_ga_weights:
                self.meta_weights.update(valid_ga_weights)
                logger.info(
                    f"[GA] Loaded GA‑optimised meta‑weights from {self.GA_PATH}: {valid_ga_weights}"
                )
            else:
                logger.warning(
                    f"[GA] Weight file {self.GA_PATH} had no matching keys – ignoring."
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"[GA] Failed to load GA weights – using defaults. Reason: {exc}")


__all__ = ["AIAgent4"]
