import os
import time 
import pickle
from collections import defaultdict
from .index_paths import THE_TOPICS, THE_SPARSE_INDEX, THE_DENSE_INDEX
from .bm25 import BM25

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher 

from feedback_methods.rocchio_feedback import *
from feedback_methods.rm3_feedback import *

K1=0.9
B=0.4


def load_pickle_if_available(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_pickle(path, payload):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(payload, file)

def load_retriever(encoder, corpus_name):
    if encoder == 'bm25':
        index_reader = LuceneIndexReader.from_prebuilt_index(THE_SPARSE_INDEX[corpus_name])
        lucene_bm25_searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[corpus_name])
        lucene_bm25_searcher.set_bm25(k1=K1, b=B)
        retriever = BM25(searcher=lucene_bm25_searcher, task=corpus_name)
        return retriever, index_reader
    else:
        # TODO: Helper that allows us to automatically figure out kwargs
        query_encoder = AutoQueryEncoder(encoder_dir=encoder, pooling='mean')
        faiss_searcher = FaissSearcher.from_prebuilt_index(THE_DENSE_INDEX[encoder][corpus_name], 
                                                        query_encoder)
        retriever = BM25(searcher=faiss_searcher, task=corpus_name)
        
        docid_dict = {retriever.searcher.docids[i]: i for i in range(len(retriever.searcher.docids))}
        return retriever, query_encoder, docid_dict

def run_bm25(bm25, 
             qids, 
             queries, 
             orig_queries,
             num_hits, 
             corpus_name, 
           #  query_generator,
             output_filename):
    start_time = time.time()
    bm25_outputs = bm25.run_search(qids, 
                                   queries, 
                                   orig_queries=orig_queries,
                                   k=num_hits, 
                                   return_passage_texts=True)
                                #   query_generator=query_generator)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time for search...", elapsed_time, "seconds")
    write_scores_to_file(all_qids=bm25_outputs['qids'],
                         all_docids=bm25_outputs['docids'], 
                         scores=bm25_outputs['bm25_scores'], 
                         output_filename=output_filename)
    print("Quickly evaluating retrieval....", flush=True)
    evaluate(corpus_name, output_filename)
    return bm25_outputs

def topk_flattened(bm25_outputs, k=100):
    qids = bm25_outputs['qids']
    # Group indices by qid
    grouped = defaultdict(list)
    for i, qid in enumerate(qids):
        grouped[qid].append(i)
    
    # Take only top-k per qid
    selected_indices = []
    for qid, idxs in grouped.items():
        selected_indices.extend(idxs[:k])
    
    # Build filtered flattened lists
    qids_new     = [bm25_outputs['qids'][i] for i in selected_indices]
    queries_new  = [bm25_outputs['queries'][i] for i in selected_indices]
    docids_new   = [bm25_outputs['docids'][i] for i in selected_indices]
    passages_new = [bm25_outputs['passage_texts'][i] for i in selected_indices]
    bm25_scores_new = [bm25_outputs['bm25_scores'][i] for i in selected_indices]
    
    return qids_new, queries_new, docids_new, passages_new, bm25_scores_new

def write_scores_to_file(all_qids, all_docids, scores, output_filename):
    output_filename = f'{output_filename}'
    with open(output_filename, 'w')  as f:
        rerank_scores = [{'qid': qid, 'docid': docid, 'score': float(score)} for qid, docid, score in zip(all_qids, all_docids, scores)]
        reranked_scores_sorted = sorted(
                rerank_scores,
                key=lambda x: (x['qid'], -x['score'])
            )
        
        rank = 0
        prev_qid = None
        for document in reranked_scores_sorted:
            qid = document["qid"]
            if qid != prev_qid:
                rank = 1
                prev_qid = qid
            else:
                rank += 1
            f.write(f'{qid} Q0 {document["docid"]} {rank} {document["score"]} rank\n')

def evaluate(corpus_name, output_filename):
    # Eval!
    if corpus_name in ['dl21', 'dl22', 'dl23']:
        qrels_name = f'{THE_TOPICS[corpus_name]}-passage'
    else:
        qrels_name = f'{THE_TOPICS[corpus_name]}'

    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {qrels_name} {output_filename}"))
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.20 {qrels_name} {output_filename}"))
    if 'dl' in corpus_name:
        print(os.system(f"python -m pyserini.eval.trec_eval -c -l 2 -m recall.20 {qrels_name} {output_filename}"))
    else:
        print(os.system(f"python -m pyserini.eval.trec_eval -c -m recall.20 {qrels_name} {output_filename}")) 

def get_bm25_vector(passage, index_reader):
    index_stats = index_reader.stats()
    num_docs = index_stats['documents']
    avg_length = index_stats['total_terms'] / num_docs

    doc_vector = get_query_vector(passage) # {term: freq}
    doc_length = sum(doc_vector.values())
    bm25_vector = {}
    for term, freq in doc_vector.items():
        # We only consider terms in our corpus...
        try:
            df, cf = index_reader.get_term_counts(term)
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            score = idf * (freq / (freq + K1 * (1 - B + B * ( doc_length  /  avg_length))))
            bm25_vector[term] = score
        except:
            continue
    
    return bm25_vector

def compute_bm25_score(query, passage, index_reader):
    # follows: https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md
    query_vector = get_query_vector(query)
    multihot_query_weights = {term: 1 for term in query_vector.keys()}
    bm25_weights = get_bm25_vector(passage, index_reader)
    score = sum({term: bm25_weights[term] \
                    for term in bm25_weights.keys() & \
                    multihot_query_weights.keys()}.values())
    return score

   
