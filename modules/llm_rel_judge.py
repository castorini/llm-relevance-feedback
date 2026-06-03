import os
from transformers import AutoTokenizer 

import torch
from vllm import LLM, SamplingParams
from .llm import BaseLLM 
from .hyde import HyDE

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

REL_JUDGE_PROMPT = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query, 
1 = represents that the passage seems related to the query but does not answer it, 
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {}
Passage: {}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""

REL_JUDGE_PROMPT_ARGUANA = """Given a query (an argument) and a passage (a potential counterargument), you must provide a score on an integer scale of 0 to 3 with the following meanings:
**0 = Non-Argument:** The passage is not a coherent argument, is completely irrelevant to the topic, or addresses a different topic entirely.
**1 = Related Argument (Not Counter):** The passage is a coherent argument related to the overall topic of the query, but it does *not* directly counter the query argument. It might support or elaborate on the same side.
**2 = Weak Counterargument:** The passage presents an argument that directly attempts to challenge or refute the query argument, but the counterpoint is weak, poorly articulated, indirect, or hidden amongst substantial irrelevant information.
**3 = Strong Counterargument (Direct Refutation):** The passage is a clear, coherent, and strong argument that directly and effectively refutes or counters the main point of the query argument.

Important Instruction: Assign category 1 if the passage is a relevant argument but does not oppose the query, category 2 if the passage attempts to oppose the query but is weak or confusing, and category 3 if the passage presents a clear and strong opposing argument. If none of the above satisfies give it category 0.

Query (Argument): {}
Passage (Potential Counterargument): {}

Split this problem into steps:
Consider the underlying stance of the query argument (Q).
Measure whether the passage directly addresses and attempts to refute the main point of the query argument (M).
Measure the coherence and strength of the passage's counter-claim (S).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""

class LLMRelevanceJudge(HyDE):
    def _process_with_vllm_rel_judge(self, prompts):
        outputs = self._generate_model_outputs(prompts)        
        all_judgements = [] 
        for i, output in enumerate(outputs):
            all_judgements.append(output.outputs[0].text)
        return all_judgements
    
    def return_prompt_judgements(self, query, doc_content, task) -> str:
        if task == 'arguana':
            chat = [
                    {'role': "user", 'content': REL_JUDGE_PROMPT_ARGUANA.format(query, doc_content)},
                    {'role': "assistant", 'content': '##final score: '},
                ]
        else:
            chat = [
                    {'role': "user", 'content': REL_JUDGE_PROMPT.format(query, doc_content)},
                    {'role': "assistant", 'content': '##final score: '},
                ]
        prompt_text = self.tokenizer.apply_chat_template(chat, continue_final_message=True, tokenize=False, enable_thinking=False)
        prompt_text = self.truncate(prompt_text, 32000 - 5)
        return prompt_text
    
    @torch.inference_mode()
    def predict(self, queries, passages, gen_task, dataset):
        if gen_task == 'passage_generation':
            return HyDE.predict(self, queries=queries, task=dataset)
        else:
            prompts = [
                self.return_prompt_judgements(query, passage, task=dataset)
                for query, passage in zip(queries, passages)
            ]
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                allowed_token_ids = [self.tokenizer.convert_tokens_to_ids(str(i)) for i in range(4)],
                skip_special_tokens=False
            )

            outputs = self._process_with_vllm_rel_judge(prompts)
        return outputs