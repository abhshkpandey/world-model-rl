CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "checkpoint_dir": "checkpoints/",
    "log_dir": "logs/",
    "num_workers": 4,
    "gradient_accumulation_steps": 2,
    "mixed_precision": True
}
