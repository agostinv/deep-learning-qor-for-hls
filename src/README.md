### Briefly Covering Source Code Layout

Pretty literal, for the most part, but I'll cover everything briefly here.

| Sub-Directory | Description |
| ------- | ----------- |
| `lightning_modules` | Necessary modules for training built in PyTorch Lightning. `codellama_embed.py` should be functional out of the box. |
| `models` | Used for custom model construction (e.g. classifying encoder w/ cross-attention, denoted as "bidirectional decoder" in files). |
| `trainers` | Scripts for training, again built on PyTorch Lightning. `codellama_embed_finetune.py` should work out of the box. Originally intended to support PEFT w/ FSDP but had some issues in getting that to work with PyTorch Lightning. |
