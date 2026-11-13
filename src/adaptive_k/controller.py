"""Controller consumes observed counts only; cannot inspect answers or gold IDs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerConfig:
    min_k: int = 3
    max_k: int = 10
    step: int = 2
    min_useful: int = 2
    low_useful_ratio: float = 0.25
    redundancy_ratio: float = 0.75

    def __post_init__(self):
        if any(type(x) is not int for x in (self.min_k, self.max_k, self.step, self.min_useful)):
            raise ValueError('K, step, and counts must be integers')
        if not 1 <= self.min_k <= self.max_k or self.step < 1 or self.min_useful < 1:
            raise ValueError('Invalid controller bounds')
        if not 0 <= self.low_useful_ratio <= 1 or not 0 <= self.redundancy_ratio <= 1:
            raise ValueError('Ratios must be in [0, 1]')


@dataclass(frozen=True)
class Feedback:
    retrieved: int
    useful: int
    new: int
    new_useful: int

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.retrieved,self.useful,self.new,self.new_useful)):
            raise ValueError('Feedback counts must be nonnegative integers')
        if self.useful > self.retrieved or self.new > self.retrieved or self.new_useful > min(self.new,self.useful):
            raise ValueError('Inconsistent feedback counts')


class DynamicController:
    def __init__(self, config=None):
        self.config = config or ControllerConfig()

    def update(self, k, feedback):
        c = self.config
        if feedback.retrieved == 0:
            target, reason = k + c.step, 'empty_results'
        elif (1 - feedback.new / feedback.retrieved >= c.redundancy_ratio
              and feedback.new_useful == 0):
            target, reason = k - c.step, 'redundant_without_new_useful_evidence'
        elif feedback.useful < c.min_useful or feedback.useful / feedback.retrieved < c.low_useful_ratio:
            target, reason = k + c.step, 'sparse_useful_evidence'
        else:
            target, reason = k, 'maintain_evidence_yield'
        selected = max(c.min_k, min(c.max_k, target))
        return selected, reason + ('_bounded' if selected != target else '')
