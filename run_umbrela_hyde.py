import os 
import math 
import argparse
import numpy as np 
import pickle 
import torch 

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher 

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

from run_hyde import generate_sparse_hyde_query_from_feedback, generate_dense_hyde_query_from_feedback
from run_prf_umbrela import get_relevant_docids, run_llm_relevance_feedback, MAX_TOP_DOCS_ALLOWED

def generate_rm3_query(qid, 
                       query, 
                       pseudo_qrels, 
                       index_reader,
                       hyde_vectors=None, 
                       hyde_scores=None):
    if qid not in pseudo_qrels:
        return query
    
    relevant_docids = get_relevant_docids(pseudo_qrels, qid)
    rel_feedback_vectors = [get_document_vector_rm3(docid, index_reader) for docid, _ in relevant_docids] 
    scores = [score for _, score in relevant_docids]

    # New Logic: Pad with HyDE if we have fewer than 8 relevant docs
    if len(rel_feedback_vectors) < 8 and hyde_vectors is not None:
        num_to_add = 8 - len(rel_feedback_vectors)
        rel_feedback_vectors += hyde_vectors[:num_to_add]
        if hyde_scores is not None:
            scores += hyde_scores[:num_to_add]

    rm3_query = rm3_feedback(query_vector=get_query_vector(query), 
                             rel_vectors=rel_feedback_vectors,
                             document_scores=scores,
                             top_fb_docs=MAX_TOP_DOCS_ALLOWED)
    return rm3_query

def generate_rocchio_query(qid, 
                           query, 
                           pseudo_qrels, 
                           index_reader,
                           negatives=False, 
                           hyde_vectors=None):
    if qid not in pseudo_qrels:
        return query
    
    relevant_docids = get_relevant_docids(pseudo_qrels, qid)
    rel_feedback_vectors = [get_document_vector_rocchio(docid, index_reader) for docid, _ in relevant_docids]  
    
    # # New Logic: Pad with HyDE if < 8
    if len(rel_feedback_vectors) < 8 and hyde_vectors is not None:
        num_to_add = 8 - len(rel_feedback_vectors)
        rel_feedback_vectors += hyde_vectors[:num_to_add]

    # Get negatives logic remains same...
    if negatives:
        irrelevant_docids = [docid for docid in pseudo_qrels[qid].keys() \
                                if pseudo_qrels[qid][docid]['relevance_label'] is not None \
                                and int(pseudo_qrels[qid][docid]['relevance_label']) == 0]
        non_rel_feedback_vectors = [get_document_vector_rocchio(docid, index_reader) for docid in irrelevant_docids]  
        gamma = 0.15
    else:
        non_rel_feedback_vectors = None
        gamma = 0

    rocchio_query = rocchio_feedback(query_vector=get_query_vector(query), 
                                     rel_vectors=rel_feedback_vectors,
                                     nrel_vectors=non_rel_feedback_vectors,
                                     gamma=gamma,
                                     top_fb_docs=MAX_TOP_DOCS_ALLOWED)
    return rocchio_query

def generate_avg_dense_prf(qid, query, pseudo_qrels, faiss_searcher, query_encoder, docid_dict, hyde_vectors):
    if qid not in pseudo_qrels: return query

    relevant_docids = get_relevant_docids(pseudo_qrels, qid)
    query_vector = query_encoder.encode([query])
    feedback_vectors = [faiss_searcher.index.reconstruct(docid_dict[doc_id]) for doc_id, _ in relevant_docids] 
    
    # New Logic: Pad to 8
    if len(feedback_vectors) < 8 and hyde_vectors is not None:
        num_to_add = 8 - len(feedback_vectors)
        feedback_vectors += hyde_vectors[:num_to_add]
    feedback_vectors = feedback_vectors[:MAX_TOP_DOCS_ALLOWED]
    assert(len(feedback_vectors) == MAX_TOP_DOCS_ALLOWED)

    all_emb = np.array([query_vector] + feedback_vectors)
    all_emb = np.mean(all_emb, axis=0)
    return all_emb.reshape((1, len(all_emb)))

def generate_rocchio_dense_prf(qid, query, pseudo_qrels, faiss_searcher, query_encoder, docid_dict, alpha, beta, hyde_vectors):
    if qid not in pseudo_qrels: return query

    relevant_docids = get_relevant_docids(pseudo_qrels, qid)
    query_vector = query_encoder.encode([query])
    feedback_vectors = [faiss_searcher.index.reconstruct(docid_dict[doc_id]) for doc_id, _ in relevant_docids] 
    
    # New Logic: Pad to 8
    if len(feedback_vectors) < 8 and hyde_vectors is not None:
        num_to_add = 8 - len(feedback_vectors)
        feedback_vectors += hyde_vectors[:num_to_add]
    
    feedback_vectors = feedback_vectors[:MAX_TOP_DOCS_ALLOWED]
    assert(len(feedback_vectors) == MAX_TOP_DOCS_ALLOWED)
    weighted_query_embs = alpha * query_vector
    weighted_mean_pos_doc_embs = beta * np.mean(np.array(feedback_vectors), axis=0)
    new_emb_q = weighted_query_embs + weighted_mean_pos_doc_embs
    return new_emb_q.reshape((1, len(new_emb_q)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LLM PRF w/ BM25')
    parser.add_argument('--model_path', required=True, help='base model path')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--corpus_name', required=True, help='dataset path')
    parser.add_argument('--hits_judged', type=int, default=20, help='search results that the LLM judges')
    parser.add_argument('--returned_hits', type=int, default=100, help='search hits after feedback reranking')
    parser.add_argument('--feedback_mechanism', type=str)
    parser.add_argument('--initial_encoder', type=str, default='BAAI/bge-base-en-v1.5', help='encoder for first pass') 
    parser.add_argument('--final_encoder', type=str, default='bm25', help='encoder for second pass') 
    parser.add_argument('--hyde_precomputed_passages', type=str, required=True, help='path to hyde generated passages') 
    parser.add_argument('--run_hyde_initial_retrieval', action='store_true')
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
        faiss_searcher_init = retriever_init.searcher

    # --- LOAD FINAL RETRIEVER (if different) ---
    if args.initial_encoder != args.final_encoder:
        retriever_elem_final = load_retriever(args.final_encoder, args.corpus_name)
        if args.final_encoder == 'bm25':
            retriever_final, index_reader_final = retriever_elem_final
        else:
            retriever_final, query_encoder_final, docid_dict_final = retriever_elem_final
            faiss_searcher_final = retriever_final.searcher
    else:
        # If they are the same, just point final to init vars
        if args.initial_encoder == 'bm25':
            retriever_final, index_reader_final = retriever_init, index_reader_init
        else:
            retriever_final = retriever_init
            query_encoder_final, docid_dict_final = query_encoder_init, docid_dict_init
            faiss_searcher_final = faiss_searcher_init

    ##################################################################
    # Run initial retrieval
    ##################################################################
    if args.run_hyde_initial_retrieval: 
        init_retrieval_name = f'{os.path.basename(args.initial_encoder)}-w-hyde_k-init-10' 
    else:
        init_retrieval_name = f'{os.path.basename(args.initial_encoder)}_k-init-100' 
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        llm_judge = LLMRelevanceJudge(base_model_name_or_path=args.model_path, num_gpus=args.num_gpus, dataset_prompt=args.corpus_name)    
    # if args.hyde_precomputed_passages is not None:
    with open(args.hyde_precomputed_passages, 'rb') as file:
        synthetic_passages = pickle.load(file)
        print("Loaded cached HyDE passages.")

    # HyDE generation relies on the nature of the initial encoder (sparse vs dense)
    if args.final_encoder == 'bm25':
        hyde_results = [generate_sparse_hyde_query_from_feedback(queries[idx], passages, args.feedback_mechanism, index_reader_final, return_vectors=True) \
                    for idx, passages in enumerate(synthetic_passages)] 
    else:
        hyde_results = [generate_dense_hyde_query_from_feedback(queries[idx], passages, args.feedback_mechanism, query_encoder_final, return_vectors=True) \
                    for idx, passages in enumerate(synthetic_passages)] 
        

    if args.run_hyde_initial_retrieval: 
        # TODO: fix this so we compute hyde results based on the correct initial retriever...
        init_retrieval_queries, hyde_vectors = zip(*hyde_results)
        init_retrieval_queries = list(init_retrieval_queries)
    else:
        _, hyde_vectors = zip(*hyde_results)
        # NOTE: For this specific method, we run standard BM25 (allows for fair comparison with other baselines)
        init_retrieval_queries = queries
        hyde_vectors = list(hyde_vectors)

    bm25_output_filename = os.path.join(args.output_folder, f'bm25_{init_retrieval_name}') 
    # Use retriever_init for first pass
    bm25_outputs = run_bm25(bm25=retriever_init, 
                            qids=qids, 
                            queries=init_retrieval_queries, 
                            orig_queries=queries,
                            num_hits=args.returned_hits, # For comparison purposes, we will only judge hits_judged docs...
                            corpus_name=args.corpus_name, 
                            output_filename=bm25_output_filename)
    #######################################
    # Run relevance feedback!
    #######################################

    if args.run_hyde_initial_retrieval: 
        feedback_filename = (
            f"precomputed_judgements/{args.corpus_name}_init_retrieval-{init_retrieval_name}_"
            f"model-{os.path.basename(args.model_path)}_k_judges-10__encoder-{os.path.basename(args.initial_encoder)}_precomputed_judgements"
        )
    else:
        feedback_filename = (
            f"precomputed_judgements/{args.corpus_name}_init_retrieval-{init_retrieval_name}_"
            f"model-{os.path.basename(args.model_path)}_k_judges-100__encoder-{os.path.basename(args.initial_encoder)}_precomputed_judgements"
        )
    print("Now running our relevance feedback...")
    pseudo_qrels = load_pickle_if_available(feedback_filename)
    if pseudo_qrels is not None:
        print("Loaded cached relevance judgments.")

        orig_lengths = {qid: len(docs) for qid, docs in pseudo_qrels.items()}
        pseudo_qrels = {
            qid: dict(sorted(docs.items(), key=lambda x: x[1]["bm25_score"], reverse=True)[:args.hits_judged])
            for qid, docs in pseudo_qrels.items()
        }
        if not all(len(docs) == args.hits_judged for docs in pseudo_qrels.values()):
            shortages = [
                (qid, len(docs), orig_lengths[qid])
                for qid, docs in pseudo_qrels.items()
                if len(docs) < args.hits_judged
            ]
            print(f"Some queries have fewer than {args.hits_judged} judged documents: {shortages}")
    else:
        pseudo_qrels = run_llm_relevance_feedback(llm_judge, bm25_outputs, args.hits_judged, args.corpus_name)  
        save_pickle(feedback_filename, pseudo_qrels)
    
    #######################################
    # Run feedback phase
    #######################################
    print("Now running our second search")
    filename = \
        f'{args.corpus_name}_init_retrieval-{init_retrieval_name}_feedback-{args.feedback_mechanism}_judge-{os.path.basename(args.model_path)}_num-judged-docs_{args.hits_judged}_final_encoder-{os.path.basename(args.final_encoder)}'
    filename = f'{filename}_w-hyde_vector-padding'
    
    rel_feedback_output_filename = os.path.join(args.output_folder, filename)
    
    # We use the FINAL encoder's components to generate the query suitable for the final search
    if args.final_encoder == 'bm25':
        if 'rm3' in args.feedback_mechanism:
            hyde_score = [
                        [compute_bm25_score(query, p, index_reader_final) for p in ps]
                        for query, ps in zip(queries, synthetic_passages)
                        ]

            expanded_queries = [
                generate_rm3_query(
                    qid, 
                    query, 
                    pseudo_qrels, 
                    index_reader_final, 
                    hyde_vectors=hyde_vectors[i], 
                    hyde_scores=hyde_score[i]
                )
                for i, (qid, query) in enumerate(zip(qids, queries))
            ]
        elif 'rocchio' in args.feedback_mechanism:
            if 'w_negatives' in args.feedback_mechanism:
                negatives=True
            else:
                negatives=False
            
            expanded_queries = [
                generate_rocchio_query(qid, query, pseudo_qrels, index_reader_final, negatives, hyde_vectors=hyde_vectors[idx])
                for idx, (qid, query) in enumerate(zip(qids, queries))
            ]
        else:
            expanded_queries = [
                generate_hyde_query(qid, query, pseudo_qrels, index_reader_final)
                for qid, query in zip(qids, queries)
            ]
    else:
        # Dense feedback gen needs final encoder components (faiss_searcher_final, etc.)
        if 'rocchio' in args.feedback_mechanism:
            expanded_queries = [generate_rocchio_dense_prf(qid, query, pseudo_qrels, faiss_searcher_final, query_encoder_final, docid_dict_final, alpha=0.4, beta=0.6, hyde_vectors=hyde_vectors[idx]) \
                                    for idx, (qid, query) in enumerate(zip(qids, queries))]
        else:
            expanded_queries = [generate_avg_dense_prf(qid, query, pseudo_qrels, faiss_searcher_final, query_encoder_final, docid_dict_final, hyde_vectors=hyde_vectors[idx]) \
                                    for idx, (qid, query) in enumerate(zip(qids, queries))]

    run_bm25(bm25=retriever_final, 
             qids=qids, 
             queries=expanded_queries, 
             orig_queries=queries,
             num_hits=args.returned_hits,
             corpus_name=args.corpus_name, 
             output_filename=rel_feedback_output_filename)
