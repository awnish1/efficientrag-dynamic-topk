"""Run A–E with one shared set of models, examples, and corpus. No training."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from adaptive_k.policy import RetrievalPolicy
from adaptive_k.evaluation import summarize

EXPERIMENTS = [('A_fixed5','fixed',5), ('B_fixed10','fixed',10),
               ('C_query','query',5), ('D_dynamic','dynamic',5), ('E_full','full',5)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def resource_report(args):
    files = {'corpus':Path(args.corpus), 'evaluation':Path(args.evaluation)}
    for name in ['labeler','filter']:
        base=Path(getattr(args,name))
        files[name+'_config']=base/'config.json'
        weights=list(base.glob('*.safetensors')) + list(base.glob('pytorch_model*.bin'))
        if not weights:files[name+'_weights']=base/'model.safetensors'
        else:
            for i,p in enumerate(sorted(weights)):files[f'{name}_weights_{i}']=p
    embedding_dir=Path(args.embeddings)
    # Require the saved embedding matrix: permits verifying index/corpus ID alignment.
    embeddings=sorted(embedding_dir.glob('passages_*'))
    if not embeddings:files['embeddings']=embedding_dir/'passages_00'
    else:
        for i,p in enumerate(embeddings):files[f'embeddings_{i}']=p
    report={'python':sys.version,'platform':platform.platform(),'paths':{},'missing':[], 'packages':{}}
    for name,p in files.items():
        report['paths'][name]={'path':str(p.resolve()),'exists':p.is_file()}
        if not p.is_file():report['missing'].append(str(p))
    for name in ['torch','transformers','tokenizers','faiss-cpu','spacy','sentencepiece','numpy','huggingface-hub']:
        try:report['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['packages'][name]=None
    return report,files


def load_rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def validate_data(corpus, dataset):
    ids=[d['id'] for d in corpus]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate corpus IDs')
    aliases={alias for id in ids for alias in id.split('//')}
    question_ids=[d['id'] for d in dataset]
    if len(question_ids)!=len(set(question_ids)):raise ValueError('Duplicate evaluation question IDs')
    for d in dataset:
        for chunk in d['decomposed_questions'].values():
            gold=f"{d['id']}-{chunk['positive_paragraph_idx']:02d}"
            if gold not in aliases:raise ValueError(f'Gold ID absent from corpus: {gold}')
            if 'positive_paragraph' not in chunk:raise ValueError('Missing positive_paragraph')
    # This preflight validation never informs the runtime policy.


def project_baseline(result):
    result=json.loads(json.dumps(result))
    for k in ['policy','initial_decision','pipeline_seconds','stop_reason']:result.pop(k,None)
    for k,v in result.items():
        if str(k).isdigit():
            for x in ['k','search_seconds','decision']:v.pop(x,None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labeler',required=True);p.add_argument('--filter',required=True)
    p.add_argument('--corpus',required=True);p.add_argument('--evaluation',required=True)
    p.add_argument('--embeddings',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=5);p.add_argument('--max-rounds',type=int,default=4)
    p.add_argument('--historical-reference', help='Saved checkpoint-8000 fixed-10 JSONL for reproduction check')
    p.add_argument('--policy-config');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--verify-baseline',action='store_true',help='Compare real fixed-5/10 traces with archived upstream loop')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.limit<1 or not 1<=args.max_rounds<=4:p.error('limit >=1 and 1<=max-rounds<=4 required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report,files=resource_report(args)
    (out/'preflight.json').write_text(json.dumps(report,indent=2))
    if report['missing']:
        print('Missing resources:\n'+'\n'.join(report['missing']))
        return 2
    if args.preflight_only:
        print('Resource files found. GPU, model loading, and matrix compatibility not yet validated.')
        return 0
    config=json.loads(Path(args.policy_config).read_text()) if args.policy_config else None
    for _,mode,k in EXPERIMENTS:RetrievalPolicy(mode,k,config)
    corpus=load_rows(args.corpus);dataset=load_rows(args.evaluation)[:args.limit]
    if not dataset:raise ValueError('Empty evaluation dataset')
    validate_data(corpus,dataset)
    if RetrievalPolicy('full',5,config).controller.config.max_k > len(corpus):
        raise ValueError('Configured max_k exceeds corpus size')
    import numpy as np
    import torch
    import pickle
    from transformers import DebertaV2Tokenizer,DebertaV2ForTokenClassification
    from efficient_rag.model import DebertaForSequenceTokenClassification
    from retrievers import Retriever
    import efficientrag_retrieve as pipeline
    if not torch.cuda.is_available():raise RuntimeError('Existing EfficientRAG inference requires CUDA; choose a Colab GPU runtime')
    if report['packages']['transformers']!='4.46.3':
        raise RuntimeError('Audited inference expects transformers 4.46.3; use the notebook setup and restart runtime')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    report.update({'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
                   'seed':args.seed,'arguments':vars(args),'corpus_rows':len(corpus),
                   'question_ids':[d['id'] for d in dataset], 'hashes':{k:digest(v) for k,v in files.items()},
                   'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src').rglob('*.py'))},
                   'experiment_order':[e[0] for e in EXPERIMENTS],
                   'policy':config, 'status':'running'})
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    # Only use trusted recovered embeddings. Read matrices to check rows, dimensions, and IDs.
    matrix_ids=[]
    for path in sorted(Path(args.embeddings).glob('passages_*')):
        with path.open('rb') as f:ids,matrix=pickle.load(f)
        if matrix.shape!=(len(ids),768) or not np.isfinite(matrix).all():raise ValueError(f'Invalid matrix: {path}')
        matrix_ids.extend(map(str,ids))
    if len(matrix_ids)!=len(set(matrix_ids)) or set(matrix_ids)!={str(d['id']) for d in corpus}:
        raise ValueError('Embedding IDs and corpus do not match; rebuild embeddings from this exact corpus')
    tokenizer=DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small')
    labeler=DebertaForSequenceTokenClassification.from_pretrained(args.labeler,token_labels=2,sequence_labels=2).cuda().eval()
    filter_model=DebertaV2ForTokenClassification.from_pretrained(args.filter,num_labels=2).cuda().eval()
    # Build a fresh in-memory index from the verified embeddings; never reuse a potentially stale FAISS index.
    retriever=Retriever(passage_path=args.corpus,passage_embedding_path=args.embeddings,
                        model_type='contriever',save_or_load_index=False)
    if len(corpus)<10:raise ValueError('A–E require at least ten passages')
    # Warm the shared pipeline equally before timing any configuration.
    with torch.inference_mode():
        list(pipeline.efficient_rag(labeler,filter_model,retriever,tokenizer,dataset[:1],top_k=10,max_rounds=1))
    torch.cuda.synchronize()
    report['retriever_revision'] = getattr(retriever.embedder.embedder.model.config, '_commit_hash', None)
    report['retriever_model'] = retriever.embedder.embedder.model_name_or_path
    report['retriever_fp16'] = retriever.embedder.embedder.fp16
    report['tokenizer'] = tokenizer.name_or_path
    verification={} 
    if args.verify_baseline:
        spec=importlib.util.spec_from_file_location('archived_baseline',ROOT/'backups/efficientrag_retrieve.upstream.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.MAX_ITER=args.max_rounds
        with torch.inference_mode():
            for k in [5,10]:
                original=list(old.efficient_rag(labeler,filter_model,retriever,tokenizer,dataset,top_k=k))
                current=list(pipeline.efficient_rag(labeler,filter_model,retriever,tokenizer,dataset,top_k=k,max_rounds=args.max_rounds))
                matched=[project_baseline(x) for x in original]==[project_baseline(x) for x in current]
                verification[str(k)]=matched
                if not matched:raise AssertionError(f'Fixed K={k} diverged from archived baseline')
    rows=[]
    historical_check=None
    for name,mode,k in EXPERIMENTS:
        start=time.perf_counter()
        with torch.inference_mode():
            results=list(pipeline.efficient_rag(labeler,filter_model,retriever,tokenizer,dataset,
                          top_k=k,mode=mode,policy_config=config,max_rounds=args.max_rounds))
        torch.cuda.synchronize()
        if name=='B_fixed10' and args.historical_reference:
            historical=load_rows(args.historical_reference)
            by_query={r['query']:r for r in historical}
            matches=[r['query'] in by_query and project_baseline(r)==project_baseline(by_query[r['query']]) for r in results]
            historical_check={'path':args.historical_reference,'sha256':digest(args.historical_reference),
                              'questions':len(results),'exact_trace_matches':sum(matches),
                              'all_match':all(matches)}
            (out/'historical_reproduction.json').write_text(json.dumps(historical_check,indent=2))
        metrics=summarize(results);metrics.update(experiment=name,status='measured',wall_seconds=time.perf_counter()-start)
        with (out/(name+'.jsonl')).open('x') as f:
            for row in results:f.write(json.dumps(row)+'\n')
        (out/(name+'.metrics.json')).write_text(json.dumps(metrics,indent=2))
        rows.append(metrics)
        print(name,json.dumps(metrics))
        if mode in ('dynamic','full') and metrics['within_question_k_changes']==0:
            print('No within-question K variation observed; do not claim meaningful dynamic allocation for this run.')
    with (out/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({**row,'k_distribution':json.dumps(row['k_distribution'])})
    report.update(status='complete',baseline_verification=verification,historical_reproduction=historical_check)
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
