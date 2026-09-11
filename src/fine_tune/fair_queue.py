"""Smooth weighted round robin with persistent fractional service credits."""
import math


class WeightedRoundRobin:
    def __init__(self, weights):
        self.current = {}
        self.update(weights)

    def update(self, weights):
        if not weights or any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights.values()):
            raise ValueError("Compute weights must be finite and positive")
        self.weights = {str(k): float(v) for k, v in weights.items()}
        # Preserve fairness across windows; reset credits only when membership changes.
        self.current = {k: self.current.get(k, 0.0) for k in self.weights}

    def choose(self, eligible=None):
        candidates = [k for k in self.weights if eligible is None or k in eligible]
        if not candidates:
            return None
        for key in candidates:
            self.current[key] += self.weights[key]
        selected = max(candidates, key=lambda k: self.current[k])
        self.current[selected] -= sum(self.weights[k] for k in candidates)
        return selected

    def allocate(self, count):
        return [self.choose() for _ in range(count)]
