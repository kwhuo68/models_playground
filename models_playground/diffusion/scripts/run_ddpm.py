import torch
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor

from models_playground import DDPM, UNet

transform = Compose(
    [
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ]
)


# Transforms
def transforms(examples):
    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["image"]
    ]
    del examples["image"]
    return examples


# Groups
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# Loading dataset
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

tf_dataset = dataset.with_transform(transforms)  # type: ignore
tf_dataset = tf_dataset.remove_columns("label")

# Dataloader
dataloader = DataLoader(
    tf_dataset["train"], batch_size=batch_size, shuffle=True  # type: ignore
)

batch = next(iter(dataloader))

model = UNet(
    dim=image_size,
    channels=channels,
    dim_mults=(
        1,
        2,
        4,
    ),
)

optimizer = Adam(model.parameters(), lr=1e-3)
epochs = 10
ddpm = DDPM(beta_start=0.0001, beta_end=0.02, timesteps=1000)

for epoch in range(epochs):
    print("Running epoch i = ", epoch)
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"]

        t = torch.randint(0, ddpm.timesteps, (batch_size,)).long()
        loss = ddpm.p_losses(model, batch, t, loss_type="l1")

        if step % 20 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
