import os
from transformers import AutoTokenizer 

import torch
from vllm import LLM, SamplingParams
from .llm import BaseLLM 
from .hyde import HyDE

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

WEB_SEARCH_MULTI_PRF = """Please write a passage to answer the question, using the provided reference passages as context.
Question: {}
Reference Passages: {}
Passage:"""


SCIFACT_MULTI_PRF = """Please write a scientific paper passage to support/refute the claim, based on the provided reference passages.
Claim: {}
Reference Passages: {}
Passage:"""


ARGUANA_MULTI_PRF = """Please write a counter argument for the passage, using the provided reference passages for context.
Passage: {}
Reference Passages: {}
Counter Argument:"""


TREC_COVID_MULTI_PRF = """Please write a scientific paper passage to answer the question, using the provided reference passages as context.
Question: {}
Reference Passages: {}
Passage:"""


FIQA_MULTI_PRF = """Please write a financial article passage to answer the question, using the provided reference passages as context.
Question: {}
Reference Passages: {}
Passage:"""


DBPEDIA_ENTITY_MULTI_PRF = """Please write a passage to answer the question, using the provided reference passages as context.
Question: {}
Reference Passages: {}
Passage:"""


TREC_NEWS_MULTI_PRF = """Please write a news passage about the topic, using the provided reference passages as context.
Topic: {}
Reference Passages: {}
Passage:"""

SCIDOCS_MULTI_PRF = """Please write a scientific paper passage that would be relevant to the following paper abstract, incorporating information from the provided reference passages.
Title: {}
Reference Passages: {}
Related Passage:"""

NQ_MULTI_PRF = """Please write a Wikipedia article to answer the question, using the provided reference passages as context.
Question: {}
Reference Passages: {}
Passage:"""

SIGNAL_1M_MULTI_PRF = """Please write a tweet that is relevant to the following news headline, using the provided reference passages as context.
Headline: {}
Reference Passages: {}
Tweet:"""

CLIMATE_FEVER_MULTI_PRF = """Please write a Wikipedia article to support/refute the claim, based on the provided reference passages.
Claim: {}
Reference Passages: {}
Passage:"""

multi_prf_prompt_dict = {
    'dl19': WEB_SEARCH_MULTI_PRF,
    'dl20': WEB_SEARCH_MULTI_PRF,

    'robust04': TREC_NEWS_MULTI_PRF,
    'dbpedia': DBPEDIA_ENTITY_MULTI_PRF,

    # Scientific and fact-checking collections
    'scifact': SCIFACT_MULTI_PRF,
    'covid': TREC_COVID_MULTI_PRF,
    'scidocs': SCIDOCS_MULTI_PRF,
    'bioasq': TREC_COVID_MULTI_PRF,
    'climate-fever': CLIMATE_FEVER_MULTI_PRF,
    'nfcorpus': TREC_COVID_MULTI_PRF,

    # QA datasets
    'nq': NQ_MULTI_PRF,
    'hotpotqa': NQ_MULTI_PRF,

    # Financial and argumentation datasets
    'fiqa': FIQA_MULTI_PRF,
    'arguana': ARGUANA_MULTI_PRF,

    # News datasets
    'news': TREC_NEWS_MULTI_PRF,

    # Tweet dataset
    'signal1m': SIGNAL_1M_MULTI_PRF,

}

class HyDE_PRF(HyDE): 
    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
    
    def return_prompt_hyde_prf(self, query, prf_passages, task) -> str:
        formatted_rag_text = "\n\n".join([f"{i+1}. {self.truncate(passage, 256)}" for i, passage in enumerate(prf_passages)])
        chat = [{'role': "user", 'content': multi_prf_prompt_dict[task].format(query, formatted_rag_text)}]
        prompt_text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return prompt_text
    
    @torch.inference_mode()
    def predict(self, queries, prf_passages, task):
        prompts = [self.return_prompt_hyde_prf(query, passage, task)  for query, passage in zip(queries, prf_passages)]  
        print(prompts[0])
        if 'gpt-oss' in self.model_name:
            # I believe reasoning tokens get included, so we will do
            # something hacky: increase max tokens then truncate...
            self.sampling_params = SamplingParams(temperature=0.7,
                                                  max_tokens=1024,
                                                  skip_special_tokens=False,
                                                  n=8)
        else:
            self.sampling_params = SamplingParams(temperature=0.7,
                                                  max_tokens=512,
                                                  skip_special_tokens=False,
                                                  n=8)
        outputs = self._process_with_vllm_hyp_documents(prompts)
        return outputs
