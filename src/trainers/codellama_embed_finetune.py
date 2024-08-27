import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy
from lightning_modules.codellama_embed import CodeLlamaEmbedder
    
from peft.utils.other import fsdp_auto_wrap_policy

from data.load_data import load_dataset, Design
from torch.utils.data import Dataset, DataLoader

from argparse import ArgumentParser, Namespace


def add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--compute-dtype", type=str, default="bfloat16")
    parser.add_argument("--output-path", type=str, default="test-output")
    return parser

# convert designs points into dataset with just source code
class HLSCodeDataset(Dataset):
    def __init__(self, designs):
        print(f"Creating raw source code dataset out of design points...")
        self.hls_code = {}
        for design in designs:
            self.hls_code[design.kernel_name] = design.src_code
        print(f"Finished creating raw source code dataset. Number of designs: {len(self.hls_code.keys())}")

    def __len__(self):
        return len(self.hls_code.keys())

    def __getitem__(self, idx):
        kernel_name = list(self.hls_code.keys())[idx]
        return self.hls_code[kernel_name]

def create_dataloader(designs, batch_size=1, shuffle=True):
    dataset = HLSCodeDataset(designs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    assert torch.cuda.device_count() > 1, "Double check that at least one GPU is exposed and available to this fine-tuning pipeline!"

    torch.cuda.empty_cache()

    train_designs = load_dataset(args.data_path)
    train_loader = create_dataloader(train_designs)
    compute_dtype = getattr(torch, args.compute_dtype)

    print(f"Loading model of name: {args.model_name} and compute_dtype: {args.compute_dtype}")
    embedder = CodeLlamaEmbedder(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        compute_dtype=compute_dtype
    )
    print(f"Model loaded. Setting up trainer...")
    
    # ensure that QLoRAs are wrapped correctly by FSDP, leaving saved state dicts as full
    # unless it becomes prohibitively slow
    #strategy = FSDPStrategy(
    #    auto_wrap_policy=fsdp_auto_wrap_policy(embedder.model),
    #    state_dict_type="full",    
    #)
    #
    #print(fsdp_auto_wrap_policy(embedder.model))

    # assumes you are using GPUs and that there are probably multiple
    # so we just call torch.cuda.device_count() directly instead of exposing
    # to the user for maximum performance
    # NOTE: fsdp with quantization may be broken, just use ddp for now
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        max_epochs=10,
        default_root_dir=args.output_path,
    )

    print(f"Trainer set up. Starting fine-tuning...")
    trainer.fit(embedder, train_loader)
    print(f"Fine-tuning complete. Model should be available at {args.output_path}")
