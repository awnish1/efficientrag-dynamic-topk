"""Gold annotations are used only here, after retrieval has completed."""
from collections import Counter


def summarize(results):
    if not results:
        raise ValueError('No results to summarize')
    gold_total = raw_found = retained_found = raw_all = retained_all = rounds = passages = 0
    search_seconds = pipeline_seconds = 0.0
    histogram = Counter()
    for result in results:
        gold = set(result['oracle'])
        if not gold:
            raise ValueError('Recall requires nonempty gold IDs')
        raw, retained = set(), set()
        for key, hop in result.items():
            if not str(key).isdigit():
                continue
            rounds += 1
            histogram[hop['k']] += 1
            search_seconds += hop['search_seconds']
            passages += len(hop['docs'])
            for doc in hop['docs']:
                ids = set(doc['id'].split('//'))
                raw.update(ids)
                if doc['label'] != '<TERMINATE>':
                    retained.update(ids)
        gold_total += len(gold)
        raw_found += len(gold & raw)
        retained_found += len(gold & retained)
        raw_all += int(gold <= raw)
        retained_all += int(gold <= retained)
        pipeline_seconds += result['pipeline_seconds']
    n = len(results)
    return {'questions':n, 'gold_documents':gold_total, 'raw_found':raw_found,
            'retained_found':retained_found, 'raw_recall':raw_found/gold_total,
            'labeler_retained_recall':retained_found/gold_total,
            'raw_all_gold_pct':100*raw_all/n, 'retained_all_gold_pct':100*retained_all/n,
            'average_rounds':rounds/n, 'average_retrieved_passages':passages/n,
            'retrieval_calls':rounds, 'average_search_seconds':search_seconds/n,
            'average_pipeline_seconds':pipeline_seconds/n,
            'k_distribution':dict(sorted(histogram.items())), 'distinct_k':len(histogram),
            'within_question_k_changes':sum(len({v['k'] for k,v in r.items() if str(k).isdigit()})>1 for r in results)}
