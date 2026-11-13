"""Exercise all five modes using fake retriever/model adapters. NOT a HotpotQA experiment."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_adaptive import load_loop,Retriever,sample
from adaptive_k.evaluation import summarize
from run_experiments import EXPERIMENTS

def main():
    out=ROOT/'results/synthetic_smoke';out.mkdir(exist_ok=False)
    questions=['Who wrote Hamlet?',
        'Were Scott Derrickson and Ed Wood of the same nationality?',
        'Which author who wrote a novel that inspired the film was born earlier than the director?']
    loop=load_loop(ROOT/'src/efficientrag_retrieve.py')
    rows=[]
    for name,mode,k in EXPERIMENTS:
        results=list(loop(None,None,Retriever(),None,[sample(q) for q in questions],top_k=k,mode=mode))
        with (out/(name+'.jsonl')).open('x') as f:
            for r in results:f.write(json.dumps(r)+'\n')
        # These metrics validate evaluator plumbing only, not quality or realistic latency.
        m=summarize(results);m.update(experiment=name,status='SYNTHETIC_NOT_RESEARCH')
        rows.append(m)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2))
    with (out/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps([{k:r[k] for k in ['experiment','status','k_distribution','within_question_k_changes']} for r in rows],indent=2))

if __name__=='__main__':main()
