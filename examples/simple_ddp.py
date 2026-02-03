from pyremote import remote, UvConfig

@remote(
    "localhost", 
    "ubuntu", 
    password="ubuntu123", 
    uv=UvConfig(path="~/.venv-3.12-v2", python_version="3.12", install_uv=True, delete_after_done=True), 
    dependencies=[
        "numpy==1.26.4", 
        "torch==2.9.1", 
        "transformers==4.57.3", 
        "accelerate",
        "datasets",
        "evaluate",
        "scikit-learn"
    ],
    install_verbose=True,
    multiprocessing=2,
    env={
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0,1"
    },
)
def compute():
    import sys
    import os
    import torch
    import torch.distributed as dist
    import evaluate
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        pipeline,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print('inside compute()', sys.version)
    print(f"[Rank {rank}/{world_size}] GPU: {torch.cuda.current_device()}")

    sms_dataset = load_dataset("sms_spam")
    sms_train_test = sms_dataset["train"].train_test_split(test_size=0.2)
    train_dataset = sms_train_test["train"]
    test_dataset = sms_train_test["test"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["sms"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    seed = 22
    train_tokenized = train_dataset.map(tokenize_function)
    train_tokenized = train_tokenized.remove_columns(["sms"]).shuffle(seed=seed)
    test_tokenized = test_dataset.map(tokenize_function)
    test_tokenized = test_tokenized.remove_columns(["sms"]).shuffle(seed=seed)

    id2label = {0: "ham", 1: "spam"}
    label2id = {"ham": 0, "spam": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_output_dir = "/tmp/sms_trainer"
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=8,
        max_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()

compute()