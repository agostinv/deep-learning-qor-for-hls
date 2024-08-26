import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

'''
    Engages in unsupervised fine-tuning, effectively, to try and generate more meaningful/less out-of-domain behavior
    when it comes to HLS pragmas. Goal is to port over the embeddings as a dataset for the final classifier. 
'''

class CodeLlamaQoREstimator(pl.LightningModule):
    def __init__(self, model_name="codellama/CodeLlama-7b-hf", lora_r=8, lora_alpha=32, lora_dropout=0.1, compute_dtype=torch.bfloat16):
        super().__init__()
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        # Load quantized CodeLlama model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare the model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear"  # Adjust based on CodeLlama's architecture
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        
    def forward(self, input_ids, attention_mask, pragmas):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["input_ids"]
        
        outputs = self(input_ids)
        
        # Custom loss function
        validity_loss = nn.BCEWithLogitsLoss()(outputs[:, 0], labels[:, 0])
        performance_loss = nn.MSELoss()(outputs[:, 1], labels[:, 1])
        utilization_loss = nn.MSELoss()(outputs[:, 2:], labels[:, 2:])
        
        total_loss = validity_loss + performance_loss + utilization_loss
        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)