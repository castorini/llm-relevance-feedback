import os 
import math 
import argparse
import numpy as np 
import torch
from collections import Counter, defaultdict
import pickle

from feedback_methods.rocchio_feedback import *
from feedback_methods.rm3_feedback import *

from modules.bm25 import BM25
from modules.hyde_prf import HyDE_PRF
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
        new_emb_q = np.mean(new_emb_q, axis=0)
        new_emb_q = new_emb_q.reshape((1, len(new_emb_q)))

    if return_vectors:
        return new_emb_q, feedback_vectors 
    return new_emb_q

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate HyDE-PRF')
    parser.add_argument('--model_path', required=True, help='base model path')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--prf_docs', type=int, default=10, help='search results that the LLM judges')
    parser.add_argument('--returned_hits', type=int, default=100, help='search hits after feedback reranking')
    parser.add_argument('--feedback_mechanism', type=str)
    # Decoupled encoders
    parser.add_argument('--initial_encoder', type=str, default='bm25', help='encoder for first pass') 
    parser.add_argument('--final_encoder', type=str, default='bm25', help='encoder for second pass') 
    
    parser.add_argument('--hyde_precomputed_passages', type=str, default=None, help='path to hyde generated passages') 
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()

    ##################################################################
    # Define and load searcher and index readers
    qids, queries = load_queries_qids(args.corpus_name)
    
    # --- LOAD INITIAL RETRIEVER ---
    retriever_elem_init = load_retriever(args.initial_encoder, args.corpus_name)
    if args.initial_encoder == 'bm25':
        retriever_init, index_reader_init = retriever_elem_init
    else:
        retriever_init, query_encoder_init, docid_dict_init = retriever_elem_init

    # --- LOAD FINAL RETRIEVER ---
    if args.initial_encoder != args.final_encoder:
        retriever_elem_final = load_retriever(args.final_encoder, args.corpus_name)
        if args.final_encoder == 'bm25':
            retriever_final, index_reader_final = retriever_elem_final
        else:
            retriever_final, query_encoder_final, docid_dict_final = retriever_elem_final
    else:
        if args.initial_encoder == 'bm25':
            retriever_final, index_reader_final = retriever_init, index_reader_init
        else:
            retriever_final = retriever_init
            query_encoder_final, docid_dict_final = query_encoder_init, docid_dict_init

    ##################################################################
    # Run initial search
    ##################################################################
  #  init_retrieval_name = f'{os.path.basename(args.initial_encoder)}_k-init-{args.prf_docs}'
    init_retrieval_name = f'{os.path.basename(args.initial_encoder)}-w-hyde_k-init-10' if args.hyde_precomputed_passages is not None \
                    else f'{os.path.basename(args.initial_encoder)}_k-init-100'  #{args.hits_judged}'
    bm25_output_filename = os.path.join(args.output_folder, f'bm25_{init_retrieval_name}')  
    
    if args.hyde_precomputed_passages is not None:
        with open(args.hyde_precomputed_passages, 'rb') as file:
            synthetic_passages = pickle.load(file)
            print("Loaded cached HyDE passages.")

        # HyDE generation relies on the nature of the initial encoder (sparse vs dense)
        if args.initial_encoder == 'bm25':
            hyde_results = [generate_sparse_hyde_query_from_feedback(queries[idx], passages, args.feedback_mechanism, index_reader_init, return_vectors=True) \
                        for idx, passages in enumerate(synthetic_passages)] 
        else:
            hyde_results = [generate_dense_hyde_query_from_feedback(queries[idx], passages, args.feedback_mechanism, query_encoder_init, return_vectors=True) \
                        for idx, passages in enumerate(synthetic_passages)] 
        

        init_retrieval_queries, _ = zip(*hyde_results)
        init_retrieval_queries = list(init_retrieval_queries)
    else:
        init_retrieval_queries = queries

    bm25_outputs = run_bm25(bm25=retriever_init, 
                            qids=qids, 
                            queries=init_retrieval_queries, 
                            orig_queries=queries,
                            num_hits=args.returned_hits, # For comparison purposes, we will only judge hits_judged docs...
                            corpus_name=args.corpus_name, 
                            output_filename=bm25_output_filename)

    qid_to_data = defaultdict(lambda: {'query': None, 'passages': []})
    for qid, query, passage in zip(bm25_outputs['qids'], bm25_outputs['queries'], bm25_outputs['passage_texts']):
        qid_to_data[qid]['query'] = query
        if len(qid_to_data[qid]['passages']) < args.prf_docs:
            qid_to_data[qid]['passages'].append(passage)

    result = [(qid, data['query'], data['passages']) for qid, data in qid_to_data.items()]
    new_qids, queries_subset, passages_subset = zip(*result)

    ##################################################################
    # Create HyDE expansions and generate new improved queries   
    first_stage_retriever_name = 'bm25_w_hyde' if args.hyde_precomputed_passages is not None \
                    else os.path.basename(args.initial_encoder) 
    precomputed_passages_name = \
        f'precomputed_passages_hyde_prf/hyde-prf_{args.corpus_name}_model-{os.path.basename(args.model_path)}_num_prf_passages-{args.prf_docs}_encoder-{first_stage_retriever_name}_precomputed_passages'
    # Try the old/default name first if k=10
    if args.prf_docs == 10:
        old_style_name = precomputed_passages_name.replace("_num_prf_passages-10", "")
        if os.path.exists(old_style_name):
            precomputed_passages_name = old_style_name
    synthetic_passages = load_pickle_if_available(precomputed_passages_name)
    if synthetic_passages is not None:
        print("Loaded cached HyDE-PRF passages.")
    else:
        hyde_prf = HyDE_PRF(base_model_name_or_path=args.model_path, 
                            num_gpus=args.num_gpus, 
                            dataset_prompt=args.corpus_name) 

        import time
        start = time.perf_counter()
        synthetic_passages = hyde_prf.predict(queries=list(queries_subset), prf_passages=list(passages_subset), task=args.corpus_name)
        end = time.perf_counter()
        total_time = end - start
        num_queries = len(queries)
        avg_time_per_query = total_time / num_queries
        print(f"Total time: {total_time:.4f}s")
        print(f"HyDE-PRF {args.prf_docs} documents, dataset={args.corpus_name}")
        print(f"Avg Time Per Query: {avg_time_per_query:.4f}s", f"Num Queries: {num_queries}")
        save_pickle(precomputed_passages_name, synthetic_passages)

    if args.final_encoder == 'bm25':
        expanded_queries = [generate_sparse_hyde_query_from_feedback(queries_subset[idx], 
                                                                     passages, 
                                                                     args.feedback_mechanism, 
                                                                     index_reader_final)
                                for idx, passages in enumerate(synthetic_passages)]  
    else:
        expanded_queries = [generate_dense_hyde_query_from_feedback(queries_subset[idx], 
                                                                    passages, 
                                                                    args.feedback_mechanism, 
                                                                    query_encoder_final)
                                for idx, passages in enumerate(tqdm(synthetic_passages))]    

    ##################################################################
    # Run final retrieval
    output_filename = \
        f'hyde-prf_init-{first_stage_retriever_name}_final-{os.path.basename(args.final_encoder)}_k-{args.prf_docs}_{args.feedback_mechanism}.trec'  
    
    os.makedirs(args.output_folder, exist_ok=True)
    hyde_output_filename = os.path.join(args.output_folder, output_filename)    

    run_bm25(bm25=retriever_final, 
             qids=new_qids,
             queries=expanded_queries, 
             orig_queries=queries_subset,
             num_hits=args.returned_hits,
             corpus_name=args.corpus_name, 
             output_filename=hyde_output_filename)
