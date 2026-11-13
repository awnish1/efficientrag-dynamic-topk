from dataclasses import asdict
from .complexity import ComplexityAnalyzer, ComplexityConfig
from .controller import DynamicController, ControllerConfig, Feedback

MODES = ('fixed', 'query', 'dynamic', 'full')


class RetrievalPolicy:
    def __init__(self, mode='fixed', top_k=10, config=None):
        if mode not in MODES or type(top_k) is not int or top_k < 1:
            raise ValueError('Invalid mode or top_k')
        config = config or {}
        if set(config) - {'complexity', 'controller'}:
            raise ValueError('Unknown configuration section')
        self.mode, self.top_k = mode, top_k
        self.analyzer = ComplexityAnalyzer(ComplexityConfig(**config.get('complexity', {})))
        self.controller = DynamicController(ControllerConfig(**config.get('controller', {})))
        c = self.controller.config
        if mode in ('dynamic', 'full') and not c.min_k <= top_k <= c.max_k:
            raise ValueError('Initial K outside controller bounds')
        if mode in ('query','full') and any(not c.min_k <= k <= c.max_k for k in self.analyzer.config.k_values):
            raise ValueError('Complexity K values outside bounds')

    def start(self, question):
        decision = self.analyzer.analyze(question) if self.mode in ('query','full') else {'k':self.top_k,'level':'fixed'}
        return decision['k'], decision

    def advance(self, k, chunks, labels, seen):
        # IDs are used for equality/deduplication only, never parsed for gold information.
        current_seen = set(seen)
        new = useful = new_useful = 0
        for chunk, label in zip(chunks, labels):
            novel = chunk['id'] not in current_seen
            good = label == '<CONTINUE>'
            new += int(novel); useful += int(good); new_useful += int(novel and good)
            current_seen.add(chunk['id'])
        feedback = Feedback(len(chunks), useful, new, new_useful)
        if self.mode in ('dynamic','full'):
            next_k, reason = self.controller.update(k, feedback)
        else:
            next_k, reason = k, 'fixed_for_question'
        return next_k, {'feedback': asdict(feedback), 'next_k':next_k, 'reason':reason}, current_seen

    def describe(self):
        return {'mode':self.mode,'top_k':self.top_k,'complexity':asdict(self.analyzer.config),
                'controller':asdict(self.controller.config)}
