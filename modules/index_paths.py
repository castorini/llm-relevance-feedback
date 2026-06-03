from pyserini.search import get_topics, get_qrels

THE_SPARSE_INDEX = {
    'dl19': 'msmarco-v1-passage-full',
    'dl20': 'msmarco-v1-passage-full',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat', 
    'nfcorpus': 'beir-v1.0.0-nfcorpus.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'nq': 'beir-v1.0.0-nq.flat',
    'bioasq': 'beir-v1.0.0-bioasq.flat',
    'signal1m': 'beir-v1.0.0-signal1m.flat',
    'climate-fever': "beir-v1.0.0-climate-fever.flat",
    }

THE_DENSE_INDEX = {'facebook/contriever': {'covid':  'beir-v1.0.0-trec-covid.contriever',
                                           'news': 'beir-v1.0.0-trec-news.contriever',
                                           'scifact': 'beir-v1.0.0-scifact.contriever',
                                           'fiqa': 'beir-v1.0.0-fiqa.contriever',
                                           'dbpedia': 'beir-v1.0.0-dbpedia-entity.contriever',
                                           'nfcorpus': 'beir-v1.0.0-nfcorpus.contriever',
                                           'robust04': 'beir-v1.0.0-robust04.contriever',
                                           'scidocs': 'beir-v1.0.0-scidocs.contriever',
                                           'arguana': 'beir-v1.0.0-arguana.contriever',
                                           'nq': 'beir-v1.0.0-nq.contriever',
                                           'bioasq': 'beir-v1.0.0-bioasq.contriever',
                                           'signal1m': 'beir-v1.0.0-signal1m.contriever',
                                           'climate-fever': "beir-v1.0.0-climate-fever.contriever",
                                          },
                    'facebook/contriever-msmarco': {
                                                    'covid': 'beir-v1.0.0-trec-covid.contriever-msmarco',
                                                    'news': 'beir-v1.0.0-trec-news.contriever-msmarco',
                                                    'scifact': 'beir-v1.0.0-scifact.contriever-msmarco',
                                                    'nfcorpus': 'beir-v1.0.0-nfcorpus.contriever-msmarco',
                                                    'fiqa': 'beir-v1.0.0-fiqa.contriever-msmarco',
                                                    'dbpedia': 'beir-v1.0.0-dbpedia-entity.contriever-msmarco',
                                                    'robust04': 'beir-v1.0.0-robust04.contriever-msmarco',
                                                    'scidocs': 'beir-v1.0.0-scidocs.contriever-msmarco',
                                                    'arguana': 'beir-v1.0.0-arguana.contriever-msmarco',
                                                    'nq': 'beir-v1.0.0-nq.contriever-msmarco',
                                                    'bioasq': 'beir-v1.0.0-bioasq.contriever-msmarco',
                                                    'signal1m': 'beir-v1.0.0-signal1m.contriever-msmarco',
                                                    'climate-fever': "beir-v1.0.0-climate-fever.contriever-msmarco",
                                                    },
                    'BAAI/bge-base-en-v1.5': {
                        'covid': 'beir-v1.0.0-trec-covid.bge-base-en-v1.5',
                        'news': 'beir-v1.0.0-trec-news.bge-base-en-v1.5',
                        'scifact': 'beir-v1.0.0-scifact.bge-base-en-v1.5',
                        'fiqa': 'beir-v1.0.0-fiqa.bge-base-en-v1.5',
                        'dbpedia': 'beir-v1.0.0-dbpedia-entity.bge-base-en-v1.5',
                        'nfcorpus': 'beir-v1.0.0-nfcorpus.bge-base-en-v1.5',
                        'robust04': 'beir-v1.0.0-robust04.bge-base-en-v1.5',
                        'scidocs': 'beir-v1.0.0-scidocs.bge-base-en-v1.5',
                        'arguana': 'beir-v1.0.0-arguana.bge-base-en-v1.5',
                        'nq': 'beir-v1.0.0-nq.bge-base-en-v1.5',
                        'bioasq': 'beir-v1.0.0-bioasq.bge-base-en-v1.5',
                        'signal1m': 'beir-v1.0.0-signal1m.bge-base-en-v1.5',
                        'climate-fever': "beir-v1.0.0-climate-fever.bge-base-en-v1.5",
                        },
                    }

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'nq': 'beir-v1.0.0-nq-test',
    'bioasq': 'beir-v1.0.0-bioasq-test',
    'signal1m': 'beir-v1.0.0-signal1m-test',
    'climate-fever': 'beir-v1.0.0-climate-fever-test',
}

def load_queries_qids(corpus_name):
    topics = get_topics(THE_TOPICS[corpus_name] if corpus_name != 'dl20' else 'dl20')
    if corpus_name in ['dl21', 'dl22', 'dl23']:
        qrels = get_qrels(f'{THE_TOPICS[corpus_name]}-passage')
    else:
        qrels = get_qrels(THE_TOPICS[corpus_name])
    test_only_qids_queries = set(qrels.keys())
    topics_qids = [(key, topics[key]['title'])  for key in topics if key in test_only_qids_queries]
    qids = [i[0] for i in topics_qids]
    queries = [i[1] for i in topics_qids]
    return qids, queries

