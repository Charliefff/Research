from trainer import GPT2Trainer

config = {
    "log_dir": "logs/final",
    "save_dir": "/logs/checkpoints/final/",
    "enocding_dict": "/data/tzeshinchen/research/gpt2/tokenizer/encoder.json",
    "decoding_dict": "/data/tzeshinchen/research/gpt2/tokenizer/decoder.json",
    "training_data": "/data/tzeshinchen/research/dataset/smiles_pad.txt",
    "device": "'cuda' if torch.cuda.is_available() else 'cpu'",
    "batch_size": 256,
    "block_size": 180,
    "max_iters": 200000,
    "learning_rate": 1e-4,
    "eval_iters": 200,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 3,
    "dropout": 0.2,
    "vocab_size": 63,
    "training_size": 1200000,
    "warnup_steps": 2000
}

if __name__ == "__main__":
    trainer = GPT2Trainer(config)
    trainer.main()
