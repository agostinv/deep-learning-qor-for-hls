import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from models.bidir_decoder import BidirectionalTransformerDecoder

class QoREstimator(pl.LightningModule):
    def __init__(self, pragma_dim, d_model=768, nhead=8, num_layers=3, lora_r=8, lora_alpha=32, lora_dropout=0.1, compute_dtype=torch.bfloat16):
        super().__init__()
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        # Load quantized CodeLlama model
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.code_embedder = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.code_embedder = prepare_model_for_kbit_training(self.code_embedder)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear"
        )
        
        # Apply LoRA to the model
        self.code_embedder = get_peft_model(self.code_embedder, peft_config)
        
        # Bidirectional Transformer Decoder (keep this in full precision)
        self.transformer_decoder = BidirectionalTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        self.final_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 6)
        )
        
    def forward(self, code_inputs, pragmas):
        # Encode code using QLoRA-adapted model
        code_outputs = self.code_embedder(**code_inputs, output_hidden_states=True)
        code_embeddings = code_outputs.hidden_states[-1][:, 0, :]
        
        # Initial pragma encoding
        pragma_embeddings = self.pragma_encoder(pragmas)
        
        # Reshape for transformer input (seq_len, batch, d_model)
        pragma_embeddings = pragma_embeddings.unsqueeze(0)
        code_embeddings_expanded = code_embeddings.unsqueeze(0)
        
        # Pass through transformer decoder
        transformed_pragmas = self.transformer_decoder(pragma_embeddings, code_embeddings_expanded)
        
        # Reshape back and concatenate
        transformed_pragmas = transformed_pragmas.squeeze(0)
        combined = torch.cat([code_embeddings, transformed_pragmas], dim=1)
        
        # Final classification
        return self.final_classifier(combined)
    
    def training_step(self, batch, batch_idx):
        inputs, pragmas, labels = batch
        outputs = self(inputs, pragmas)
        
        # Custom loss function
        validity_loss = nn.BCEWithLogitsLoss()(outputs[:, 0], labels[:, 0])
        performance_loss = nn.MSELoss()(outputs[:, 1], labels[:, 1])
        utilization_loss = nn.MSELoss()(outputs[:, 2:], labels[:, 2:])
        
        total_loss = validity_loss + performance_loss + utilization_loss
        self.log('train_loss', total_loss)
        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)