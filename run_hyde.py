import os 
import math 
import argparse
import numpy as np 
import torch
import pickle 

from feedback_methods.rocchio_feedback import *
from feedback_methods.rm3_feedback import *

from modules.bm25 import BM25
from modules.hyde import HyDE
from modules.llm_rel_judge import LLMRelevanceJudge
from modules.helpers import (
    load_retriever,
    run_bm25,
    topk_flattened,
    compute_bm25_score,
    load_pickle_if_available,
    save_pickle,
)
from modules.index_paths import THE_SPARSE_INDEX, THE_DENSE_INDEX, load_queries_qids

from tqdm import tqdm 

K1=0.9
B=0.4

def generate_sparse_hyde_query_from_feedback(query, 
                                             synthetic_passages, 
                                             feedback_mechanism, 
                                             index_reader, 
                                             return_vectors=False):
    synthetic_passage_vectors = [get_document_vector_hyde(p, index_reader) for p in synthetic_passages]
    query_vector = get_query_vector(query)
    if feedback_mechanism == 'rocchio':
        q_new = rocchio_feedback(query_vector=query_vector, 
                                 rel_vectors=synthetic_passage_vectors,
                                 nrel_vectors=None,
                                 gamma=0)
    elif feedback_mechanism == 'rm3':
        # For simplicity and to follow Anserini, setting document scores as the BM25 score
        document_scores = [compute_bm25_score(query, p, index_reader) for p in synthetic_passages]
        q_new = rm3_feedback(query_vector=query_vector,
                             rel_vectors=synthetic_passage_vectors,
                             document_scores=document_scores)    
    else:
        raise ValueError(f"Unsupported sparse feedback mechanism: {feedback_mechanism}")
    if return_vectors:
        return q_new, synthetic_passage_vectors 
    return q_new

def generate_dense_hyde_query_from_feedback(query, 
                                            synthetic_passages, 
                                            feedback_mechanism, 
                                            query_encoder, 
                                            return_vectors=False):
    
    feedback_vectors = [query_encoder.encode(p) for p in synthetic_passages]
    if feedback_mechanism == 'rocchio':
        alpha, beta = 0.4, 0.6
        query_vector = query_encoder.encode([query])
        weighted_query_embs = alpha * query_vector
        weighted_mean_pos_doc_embs = beta * np.mean(np.array(feedback_vectors), axis=0)
        new_emb_q = weighted_query_embs + weighted_mean_pos_doc_embs
        new_emb_q = new_emb_q.reshape((1, len(new_emb_q)))
    else:
        query_vector = query_encoder.encode([query])
        new_emb_q = np.array([query_vector] + feedback_vectors)
        print(new_emb_q.shape)
        new_emb_q = np.mean(new_emb_q, axis=0)
        new_emb_q = new_emb_q.reshape((1, len(new_emb_q)))

    if return_vectors:
        return new_emb_q, feedback_vectors 
    return new_emb_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate HyDE')
    parser.add_argument('--model_path', required=True, help='base model path')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--returned_hits', type=int, default=100, help='search hits after feedback reranking')
    parser.add_argument('--feedback_mechanism', type=str)
    parser.add_argument('--encoder', type=str, default='bm25', help='encoder name or bm25') 
    parser.add_argument('--precomputed_passages', type=str, default=None, help='optional path to cached HyDE generations')
    parser.add_argument('--save_precomputed_passages', action='store_true', help='persist generated HyDE passages to --precomputed_passages')
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    ##################################################################
    # Define and load searcher and index readers
    qids, queries = load_queries_qids(args.corpus_name)
    retriever_elem = load_retriever(args.encoder, args.corpus_name)
    if args.encoder == 'bm25':
        retriever, index_reader = retriever_elem
    else:
        retriever, query_encoder, docid_dict = retriever_elem
        faiss_searcher = retriever.searcher
    ##################################################################
    # Create HyDE expansions and generate new improved queries
    output_filename = \
        f'hyde_model-{os.path.basename(args.model_path)}_feedback-{args.feedback_mechanism}_{os.path.basename(args.encoder)}.trec'    
    
    synthetic_passages = load_pickle_if_available(args.precomputed_passages)
    if synthetic_passages is not None:
        print("Loaded cached HyDE passages.")
    else:
        hyde = HyDE(base_model_name_or_path=args.model_path, 
            num_gpus=args.num_gpus, 
            dataset_prompt=args.corpus_name) 
        import time
        start = time.perf_counter()
        synthetic_passages = hyde.predict(queries=queries, task=args.corpus_name)
        end = time.perf_counter()
        total_time = end - start
        num_queries = len(queries)
        avg_time_per_query = total_time / num_queries
        print(f"Total time: {total_time:.4f}s")
        print(f"HyDE documents, dataset={args.corpus_name}")
        print(f"Avg Time Per Query: {avg_time_per_query:.4f}s", f"Num Queries: {num_queries}")
        if args.precomputed_passages is not None and args.save_precomputed_passages:
            save_pickle(args.precomputed_passages, synthetic_passages)
    
    if args.encoder == 'bm25':
        expanded_queries = [generate_sparse_hyde_query_from_feedback(queries[idx], 
                                                                     passages, 
                                                                     args.feedback_mechanism, 
                                                                     index_reader)
                                for idx, passages in enumerate(synthetic_passages)]     
    else:
        expanded_queries = [generate_dense_hyde_query_from_feedback(queries[idx], 
                                                                    passages, 
                                                                    args.feedback_mechanism, 
                                                                    query_encoder)
                                for idx, passages in enumerate(tqdm(synthetic_passages))]    
    ##################################################################
    # Run retrieval
    os.makedirs(args.output_folder, exist_ok=True)
    hyde_output_filename = os.path.join(args.output_folder, output_filename)    
    # UPDATED: passed `retriever` instead of `bm25`
    run_bm25(bm25=retriever, 
             qids=qids, 
             queries=expanded_queries, 
             orig_queries=queries,
             num_hits=args.returned_hits,
             corpus_name=args.corpus_name, 
             output_filename=hyde_output_filename)
