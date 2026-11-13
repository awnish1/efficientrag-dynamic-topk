import ast
import copy
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from adaptive_k.policy import RetrievalPolicy
from adaptive_k.complexity import ComplexityAnalyzer, ComplexityConfig
from adaptive_k.controller import DynamicController, ControllerConfig, Feedback
from adaptive_k.evaluation import summarize


def load_loop(path, label=None, filtering=None):
    """Execute the real loop AST with fake model adapters; no duplicated loop implementation."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'efficient_rag')
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    env = {'MAX_ITER':4, 'spacy':SimpleNamespace(load=lambda _:None), 'tqdm_rich':lambda x:x,
           'CONTINUE_TAG':'<CONTINUE>', 'FINISH_TAG':'<FINISH>', 'TERMINATE_TAG':'<TERMINATE>',
           'RetrievalPolicy':RetrievalPolicy, 'time':time,
           'label_info': label or (lambda a,b,c,q,texts: (texts, ['<CONTINUE>']*len(texts))),
           'filter_query': filtering or (lambda a,b,c,q,infos: (' | '.join(infos), q+' next'))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),str(path),'exec'),env)
    return env['efficient_rag']


class Retriever:
    def __init__(self, empty=False):
        self.calls=[]; self.empty=empty
    def search(self, query, top_k):
        self.calls.append((query,top_k))
        return [[] if self.empty else [{'id':f'q-{i:02}', 'title':str(i),'text':f'passage {i}'} for i in range(top_k)]]


def sample(question='Who wrote Hamlet?'):
    return {'id':'q','question':question,'answer':'not used',
            'decomposed_questions':{'0':{'positive_paragraph_idx':0,'positive_paragraph':'not used'}}}


def baseline_projection(r):
    r=copy.deepcopy(r)
    for k in ['policy','initial_decision','pipeline_seconds','stop_reason']:r.pop(k,None)
    for k,v in r.items():
        if str(k).isdigit():
            for x in ['k','search_seconds','decision']:v.pop(x,None)
    return r


class Tests(unittest.TestCase):
    def test_complexity(self):
        a=ComplexityAnalyzer()
        for q in ['', 'Who wrote Hamlet?', 'Were Scott Derrickson and Ed Wood of the same nationality?',
                  'Which author who wrote a novel that inspired the film was born earlier than the director?']:
            self.assertIn(a.analyze(q)['k'],(3,5,8))
        self.assertEqual(a.analyze('Who wrote Hamlet?')['k'],3)
        self.assertEqual(a.analyze('Which author who wrote a novel that inspired the film was born earlier than the director?')['k'],8)

    def test_policy_validation(self):
        for kwargs in [dict(min_k=0),dict(max_k=1),dict(step=0),dict(low_useful_ratio=2)]:
            with self.assertRaises(ValueError):ControllerConfig(**kwargs)
        with self.assertRaises(ValueError):ComplexityConfig(k_values=(8,5,3))
        with self.assertRaises(ValueError):RetrievalPolicy('typo')

    def test_feedback_and_bounds(self):
        c=DynamicController()
        self.assertEqual(c.update(5,Feedback(5,1,5,1))[0],7)
        self.assertEqual(c.update(5,Feedback(5,3,5,3))[0],5)
        self.assertEqual(c.update(5,Feedback(5,3,0,0))[0],3)
        self.assertEqual(c.update(10,Feedback(0,0,0,0))[0],10)
        for k in range(3,11):
            for f in [Feedback(0,0,0,0),Feedback(5,0,0,0),Feedback(5,5,5,5)]:
                self.assertTrue(3<=c.update(k,f)[0]<=10)

    def test_fixed_matches_original(self):
        old=load_loop(ROOT/'backups/efficientrag_retrieve.upstream.py')
        new=load_loop(ROOT/'src/efficientrag_retrieve.py')
        for k in [3,5,10]:
            a,b=Retriever(),Retriever()
            expected=list(old(None,None,a,None,[sample()],top_k=k))
            actual=list(new(None,None,b,None,[sample()],top_k=k))
            self.assertEqual(expected,[baseline_projection(r) for r in actual])
            self.assertEqual(a.calls,b.calls)
            self.assertEqual(len(b.calls),4)

    def test_stop_preserved(self):
        def labels(a,b,c,q,texts):return texts,['<TERMINATE>']*len(texts)
        def fail(*args):raise AssertionError('Filter must not run without CONTINUE')
        new=load_loop(ROOT/'src/efficientrag_retrieve.py',labels,fail)
        for mode in ['fixed','query','dynamic','full']:
            ret=Retriever();r=list(new(None,None,ret,None,[sample()],top_k=5,mode=mode))[0]
            self.assertEqual(len(ret.calls),1)
            self.assertFalse(r[0]['decision']['applied'])

    def test_empty_and_missing_scores(self):
        new=load_loop(ROOT/'src/efficientrag_retrieve.py')
        for empty in [True,False]:
            ret=Retriever(empty);r=list(new(None,None,ret,None,[sample()],top_k=5,mode='full'))[0]
            self.assertGreaterEqual(len(ret.calls),1)
            if empty:self.assertEqual(r[0]['docs'],[])

    def test_filter_output_and_max_rounds(self):
        # Empty Filter output did NOT stop upstream: preserve that behavior, including final filter call.
        calls=[]
        def filtering(*args):calls.append(1);return 'input',''
        new=load_loop(ROOT/'src/efficientrag_retrieve.py',filtering=filtering)
        ret=Retriever();list(new(None,None,ret,None,[sample()],top_k=5))
        self.assertEqual(len(calls),4);self.assertEqual(ret.calls[1][0],'')
        with self.assertRaises(ValueError):list(new(None,None,ret,None,[sample()],max_rounds=5))

    def test_actual_k_changes(self):
        new=load_loop(ROOT/'src/efficientrag_retrieve.py')
        ret=Retriever();r=list(new(None,None,ret,None,[sample()],top_k=5,mode='dynamic'))[0]
        self.assertEqual([k for q,k in ret.calls],[5,5,3,3])
        self.assertEqual([r[i]['k'] for i in range(4)],[5,5,3,3])

    def test_no_oracle_decisions(self):
        new=load_loop(ROOT/'src/efficientrag_retrieve.py')
        a=sample();b=sample();b['answer']='OTHER';b['id']='changed'
        b['decomposed_questions']['0']['positive_paragraph']='unrelated'
        traces=[]
        for s in [a,b]:
            ret=Retriever();list(new(None,None,ret,None,[s],top_k=5,mode='full'));traces.append(ret.calls)
        self.assertEqual(*traces)

    def test_per_question_reset(self):
        new=load_loop(ROOT/'src/efficientrag_retrieve.py');ret=Retriever()
        rows=list(new(None,None,ret,None,[sample(),sample()],top_k=5,mode='full'))
        self.assertEqual(rows[0][0]['decision']['feedback'],rows[1][0]['decision']['feedback'])

    def test_metrics_merged_ids(self):
        result={'oracle':['a','b'], 'pipeline_seconds':1, 0:{'k':3,'search_seconds':0.1,
                'docs':[{'id':'a//b','label':'<CONTINUE>'},{'id':'c','label':'<TERMINATE>'}]}}
        m=summarize([result]);self.assertEqual(m['raw_recall'],1);self.assertEqual(m['labeler_retained_recall'],1)
        self.assertEqual(m['average_retrieved_passages'],2)


if __name__=='__main__':unittest.main()
