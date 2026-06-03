import os
from transformers import AutoTokenizer 

import torch
from vllm import LLM, SamplingParams
from .llm import BaseLLM 

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
Passage:"""


SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""


ARGUANA = """Please write a counter argument for the passage.
Passage: {}
Counter Argument:"""


TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""


FIQA = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""


DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
Passage:"""


TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
Passage:"""

SCIDOCS = """Please write a scientific paper passage that would be relevant to the following paper abstract.
Title: {}
Related Passage:"""

NQ = """Please write a Wikipedia article to answer the question.
Question: {}
Passage:"""

SIGNAL_1M = """Please write a tweet that is relevant to the following news headline.
Headline: {}
Tweet:"""

CLIMATE_FEVER = """Please write a Wikipedia article to support/refute the claim.
Claim: {}
Passage:"""


prompt_dict = {
    'dl19': WEB_SEARCH,
    'dl20': WEB_SEARCH,
    'robust04': TREC_NEWS,
    'dbpedia': DBPEDIA_ENTITY,

    # Scientific and fact-checking collections
    'scifact': SCIFACT,
    'covid': TREC_COVID,
    'scidocs': SCIDOCS,
    'bioasq': TREC_COVID,
    'climate-fever': CLIMATE_FEVER,
    'nfcorpus': TREC_COVID,

    # QA datasets
    'nq': NQ,

    # Financial and argumentation datasets
    'fiqa': FIQA,
    'arguana': ARGUANA,

    # News datasets
    'news': TREC_NEWS,

    # Tweet dataset
    'signal1m': SIGNAL_1M,
}

class HyDE(BaseLLM):    
    def _extract_gptoss_content(self, text):
        passage_text = text.split("<|start|>assistant<|channel|>final<|message|>")[-1].strip()
        passage_text = self.truncate(passage_text, 512)
        return passage_text

    def _process_with_vllm_hyp_documents(self, prompts):
        outputs = self._generate_model_outputs(prompts)  
        all_passages = []
        for i, output in enumerate(outputs):
            sample_passages = [] 
            for sample in output.outputs:
                text = sample.text
                if 'gpt-oss' in self.model_name:
                    text = self._extract_gptoss_content(text)
                sample_passages.append(text)
            all_passages.append(sample_passages)
        return all_passages

    def return_prompt_hyde(self, query, task) -> str:
        chat = [{'role': "user", 'content': prompt_dict[task].format(query)}]
        prompt_text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return prompt_text
    
    @torch.inference_mode()
    def predict(self, queries, task):
        prompts = [self.return_prompt_hyde(query, task) for query in queries]  
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