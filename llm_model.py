import re
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from serpersearch import run_google_search


class FTMistral:
    def __init__(self):
        adapter_model = "NikyParfenov/mistral-gutenberg-books-finetune"
        base_model_id = "mistralai/Mistral-7B-v0.1"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            add_bos_token=True,
            trust_remote_code=True,
        )
        self.ft_model = PeftModel.from_pretrained(base_model, adapter_model)


    def run_llm(self, prompt, use_google_search=True):

        rag = '[]'
        if use_google_search:
            try:
                rag = run_google_search(prompt)
                logger.info(f'Google search has been finished!')
            except Exception as e:
                logger.error(f'Google search error: {e}')
                
        logger.info(f'Run LLM inference')
        eval_prompt = "### {rag}\n### Question: {input}\n### Answer:".format(rag=rag, input=prompt)
        model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")

        response = self.tokenizer.decode(self.ft_model.generate(**model_input, max_new_tokens=512)[0], 
                                    skip_special_tokens=True, 
                                    pad_token_id=self.tokenizer.eos_token_id)
        logger.info(f'LLM completion has been finished!')
        try:
            return re.search(rf'(?<=### Answer: )[^\n\n]*', response, flags=re.I).group(0)
        except:
            return response


llm = FTMistral()
