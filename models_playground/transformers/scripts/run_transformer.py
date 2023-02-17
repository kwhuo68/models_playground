import torch.nn as nn
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from models_playground import Transformer

# Loading dataset
dataset = load_dataset("imdb")
num_tokens = 30000
seq_length = 512
batch_size = 16


# Tokenize the text data
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        max_length=seq_length,
        padding="max_length",
        truncation=True,
    ),
    batched=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset.set_format(  # type: ignore
    "torch", columns=["input_ids", "label", "attention_mask"]
)
tf_dataset = dataset

# Dataloader
dataloader = DataLoader(
    tf_dataset["train"],  # type: ignore
    batch_size=batch_size,
    collate_fn=data_collator,
    shuffle=True,
)

# Model
embedding_dim = 256
num_heads = 8
depth = 6

model = Transformer(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    depth=depth,
    seq_length=seq_length,
    num_tokens=num_tokens,
)
optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 10

for epoch in range(epochs):
    print("Running epoch i = ", epoch)
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        logits = model(input_ids)
        target = input_ids[:, 0]

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, target)

        if step % 20 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
