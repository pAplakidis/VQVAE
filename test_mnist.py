#!/usr/bin/env python3
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.model import VQVAE

BS = 2
EPOCHS = 10


transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
        ])

def train(model, loader, l2_loss, optim):
  model.train()

  for i_batch, sample in enumerate((t := tqdm(loader))):
    X = sample[0].to(device)
    Y = X.clone()

    optim.zero_grad()
    out, quantize_loss = model(X)

    # FIXME: out.shape != Y.shape
    print(out.shape, Y.shape)

    loss = l2_loss(out, Y) + quantize_loss
    loss.backward()
    optim.step()

    t.set_description(f"loss = {loss.item():.2f}")

def test(model, loader):
  pass


if __name__ == "__main__":
  device = torch.device("cpu")
  print(f"[+] Using device: {device}")

  train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)

  test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)
  test_loader = DataLoader(test_set, batch_size=BS, shuffle=True)

  model = VQVAE(1, 1).to(device)

  optim = torch.optim.Adam(model.parameters(), lr=1e-4)
  l2_loss = nn.MSELoss()
  for epoch in range(EPOCHS):
    print(f"[*] Epoch {epoch + 1}")
    train(model, train_loader, l2_loss, optim)
    test(model, test_loader)
