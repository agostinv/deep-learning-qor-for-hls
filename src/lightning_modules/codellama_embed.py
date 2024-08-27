import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

'''
    Engages in unsupervised fine-tuning, effectively, to try and generate more meaningful/less out-of-domain behavior
    when it comes to HLS pragmas. Goal is to port over the embeddings as a dataset for the final classifier. 
'''

class CodeLlamaEmbedder(pl.LightningModule):
    def __init__(self, model_name="codellama/CodeLlama-7b-hf", lora_r=8, lora_alpha=32, lora_dropout=0.1, compute_dtype=torch.bfloat16):
        super().__init__()
        self.model_name = model_name
        self.model = None

        # Configure 4-bit quantization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=compute_dtype,
        )
        
        # Configure LoRA
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear"
        )
        
    # should reduce memory "peakiness" by setting up the model after passing any strategies to the trainer
    # currently isn't easy to enable due to the way PyTorch Lightning handles model setup with QLoRA
    # NOTE: fine so long as DDP is being used, otherwise initialization functionality needs to be added back due to
    #       the aforementioned problems with QLoRA and FSDP
    def configure_model(self):
        if self.model is not None:
            return
        
        # Load quantized CodeLlama model, assume PyTorch Lightning Trainer will manage sharding and FSDP
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
        )
        # Prepare the model for k-bit training, sort of necessary for PEFT
        self.model = prepare_model_for_kbit_training(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, self.peft_config)

        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs

    def training_step(self, batch, batch_idx):
        print(len(batch))

        # may break for FSDP, but fine for DDP while quantization isn't working for FSDP
        encoding = self.tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = encoding["input_ids"].to(device=self.model.device)
        attention_mask = encoding["attention_mask"].to(device=self.model.device)

        # passing attention mask to ensure padding is properly managed
        outputs = self(input_ids, attention_mask)
        logits = outputs.logits

        # Shift the labels for language modeling objective, unsupervised
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Flatten the tokens, nn.CrossEntropy(...) expects M x V where V is the vocab size
        # and M is (bsz x N) after flattening with view(...)
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        self.log('train_loss', loss)
        return loss       

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)
