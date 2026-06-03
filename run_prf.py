import os 
import argparse
from tqdm import tqdm 

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from feedback_methods.rocchio_feedback import *
from feedback_methods.rm3_feedback import *

from run_prf_umbrela import \
    generate_rm3_query, \
    generate_rocchio_query,  \
    generate_avg_dense_prf, \
    generate_rocchio_dense_prf 

from modules.index_paths import THE_SPARSE_INDEX, load_queries_qids
from modules.helpers import load_retriever, run_bm25, topk_flattened, compute_bm25_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Custom PRF w/ BM25')
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--encoder', type=str, default='', help='encoder for first pass') 
    parser.add_argument('--feedback_documents', type=int, default=10, help='search results that PRF uses')
    parser.add_argument('--returned_hits', type=int, default=100, help='search results for feedback reranking')
    parser.add_argument('--feedback_mechanism', default=None, type=str)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    ##################################################################
    # Define and load searcher and index readers
    qids, queries = load_queries_qids(args.corpus_name)
    # Load retriever
    retriever_elem = load_retriever(args.encoder, args.corpus_name)
    
    if args.encoder == 'bm25':
        retriever, index_reader = retriever_elem
    else:
        retriever, query_encoder, docid_dict = retriever_elem
        faiss_searcher = retriever.searcher
    
    output_filename = os.path.join(args.output_folder, f'{os.path.basename(args.encoder)}.trec')
    ##################################################################
    # Run retrieval
    os.makedirs(args.output_folder, exist_ok=True)
    # For now, it's the same, but in case someone wants to use HyDE, etc, we keep. 
    init_retrieval_queries = queries
    bm25_outputs = run_bm25(bm25=retriever, 
                            qids=qids, 
                            queries=init_retrieval_queries, 
                            orig_queries=queries,
                            num_hits=args.returned_hits, # For comparison purposes, we will only judge hits_judged docs...
                            corpus_name=args.corpus_name, 
                            output_filename=output_filename)
    if args.feedback_mechanism is not None:
        # We will now mimic our llm-judge code, except now the top-k are all relevant! Rest is 0. 
        qids_topk, queries_topk, docids_topk, passages_topk, bm25_scores_new = \
                topk_flattened(bm25_outputs, k=args.feedback_documents)
        pseudo_qrels = {} 
        for qid, docid, scores in zip(qids_topk,
                                                 docids_topk, 
                                                #  relevance_labels,
                                                 bm25_scores_new):
            if qid in pseudo_qrels:
                pseudo_qrels[qid][docid] = {'relevance_label': 1, 
                                            'bm25_score': scores}
            else:
                pseudo_qrels[qid] = {docid: {'relevance_label': 1, 
                                            'bm25_score': scores}}
        # Fill out rest of "pseudo_qrels" w/ None. 
        for qid, docid, scores in zip(bm25_outputs['qids'], bm25_outputs['docids'], bm25_outputs['bm25_scores']):
            if docid not in pseudo_qrels[qid]:
                pseudo_qrels[qid][docid] = {'relevance_label': None, 
                                            'bm25_score': scores}
        
        # Now we can just use what we have created before! 
        #######################################
        # Run feedback phase
        #######################################
        print("Now running our second search")
        rel_feedback_output_filename = os.path.join(args.output_folder, f'{os.path.basename(args.encoder)}_{args.feedback_mechanism}_python_impl.trec')
        # We use the FINAL encoder's components to generate the query suitable for the final search
        if args.encoder == 'bm25':
            if 'rocchio' in args.feedback_mechanism:
                if 'w_negatives' in args.feedback_mechanism:
                    negatives=True
                else:
                    negatives=False
                
                expanded_queries = [
                    generate_rocchio_query(qid, query, pseudo_qrels, index_reader, negatives, hyde_vectors=None)
                    for idx, (qid, query) in enumerate(zip(qids, queries))
                ]
            else:
                # TODO: THIS NEEDS TO BE FIXED.... See w_hyde_padding code file for an idea...
                expanded_queries = [
                    generate_rm3_query(
                        qid, 
                        query, 
                        pseudo_qrels, 
                        index_reader,
                        hyde_vectors=None, 
                        hyde_scores=None,
                    )
                    for i, (qid, query) in enumerate(zip(qids, queries))
                ]
        else:
            if 'rocchio' in args.feedback_mechanism:
                expanded_queries = [generate_rocchio_dense_prf(qid, 
                                                               query, 
                                                               pseudo_qrels, 
                                                               faiss_searcher, 
                                                               query_encoder, 
                                                               docid_dict, 
                                                               alpha=0.4, 
                                                               beta=0.6, 
                                                               hyde_vectors=None) \
                                        for idx, (qid, query) in enumerate(zip(qids, queries))]
            else:
                expanded_queries = [generate_avg_dense_prf(qid, 
                                                           query, 
                                                           pseudo_qrels, 
                                                           faiss_searcher, 
                                                           query_encoder, 
                                                           docid_dict, 
                                                           hyde_vectors=None) \
                                        for idx, (qid, query) in enumerate(zip(qids, queries))]

        run_bm25(bm25=retriever, 
                 qids=qids, 
                 queries=expanded_queries, 
                 orig_queries=queries,
                 num_hits=args.returned_hits,
                 corpus_name=args.corpus_name, 
                 output_filename=rel_feedback_output_filename)
