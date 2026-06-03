import os
from transformers import AutoTokenizer 

import torch
from vllm import LLM, SamplingParams

class BaseLLM():
    def __init__(
        self,
        base_model_name_or_path: str,
        batch_size: int = 999999999999,
        context_size: int = 32000,
        max_output_tokens: int = 8192,
        fp_options: str = "float16",
        num_gpus: int = 1,
        device: str = "cuda",
        dataset_prompt: str = 'default',
    ):

        self.model_name = base_model_name_or_path
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.num_gpus = num_gpus
        self.device = device
        self.dataset_prompt = dataset_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            dtype='bfloat16',
            gpu_memory_utilization=0.9,
            enforce_eager=True,
        )
    
    def _generate_model_outputs(self, prompts):
        return self.model.generate(prompts, self.sampling_params) 
    
    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])