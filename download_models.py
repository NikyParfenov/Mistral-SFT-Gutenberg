from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os


def download_model(model_path='base_model/', ft_model_path='ft_model/'):
    """Download a Hugging Face model and tokenizer to the specified directory"""

    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)
        
    if not os.path.exists(ft_model_path):
        # Create the directory
        os.makedirs(ft_model_path)

    adapter_model = "NikyParfenov/mistral-gutenberg-books-finetune"
    base_model_id = "mistralai/Mistral-7B-v0.1"

    def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    device = get_device_map()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config, 
        device_map=device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_model)

    base_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(ft_model_path)

download_model()