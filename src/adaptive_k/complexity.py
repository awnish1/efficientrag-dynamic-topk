"""Cheap, deterministic surface features; these are hypotheses, not calibrated scores."""
import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ComplexityConfig:
    k_values: tuple = (3, 5, 8)
    length_threshold: int = 16
    long_threshold: int = 25
    medium_score: int = 1
    high_score: int = 3

    def __post_init__(self):
        if len(self.k_values) != 3 or any(type(k) is not int or k < 1 for k in self.k_values):
            raise ValueError('k_values must contain three positive integers')
        if list(self.k_values) != sorted(set(self.k_values)):
            raise ValueError('k_values must be strictly increasing')
        if not 0 < self.length_threshold <= self.long_threshold:
            raise ValueError('Invalid length thresholds')
        if not 0 < self.medium_score < self.high_score:
            raise ValueError('Invalid score thresholds')


class ComplexityAnalyzer:
    def __init__(self, config=None):
        self.config = config or ComplexityConfig()

    def analyze(self, question):
        words = re.findall(r"\b[\w'-]+\b", question)
        text = question.lower()
        # Capitalized spans are an entity proxy, not an NER claim. Skip question opener.
        spans = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ' '.join(words[1:]))
        relational = len(re.findall(r'\b(?:whose|that|which|who)\b', ' '.join(words[1:]).lower()))
        comparison = bool(re.search(r'\b(?:both|same|different|earlier|later|older|younger|more|less|compare|comparison)\b', text))
        c = self.config
        score = (int(len(words) >= c.length_threshold) + int(len(words) >= c.long_threshold)
                 + int(len(spans) >= 2) + int(relational > 0) + int(comparison))
        level = 2 if score >= c.high_score else 1 if score >= c.medium_score else 0
        return {'k': c.k_values[level], 'score': score, 'level': ['low','medium','high'][level],
                'features': {'words': len(words), 'capitalized_spans': len(spans),
                             'relational_markers': relational, 'comparison': comparison}}
