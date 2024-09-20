#!/usr/bin/env python3
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VQVAE

LOAD = True

BS = 256
N_WORKERS = 8
EPOCHS = 10

transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
        ])

def train(model, loader, l2_loss, optim, losses):
  model.train()

  for i_batch, sample in enumerate((t := tqdm(loader))):
    X = sample[0].to(device)
    Y = X.clone()

    optim.zero_grad()
    out, quantize_loss = model(X)

    loss = l2_loss(out, Y) + quantize_loss
    loss.backward()
    optim.step()

    losses.append(loss.item())
    t.set_description(f"training loss = {loss.item():.2f}")

def test(model, loader, l2_loss, losses):
  model.eval()

  with torch.no_grad():
    for i_batch, sample in enumerate((t := tqdm(loader))):
      X = sample[0].to(device)
      Y = X.clone()

      out, quantize_loss = model(X)
      loss = l2_loss(out, Y) + quantize_loss

      losses.append(loss.item())
      t.set_description(f"eval loss = {loss.item():.2f}")

def show_results(model, loader):
  with torch.no_grad():
    for i_batch, sample in enumerate((t := tqdm(loader))):
      X = sample[0].to(device)
      out, quantize_loss = model(X)

      img = X.squeeze(0).squeeze(0).cpu().numpy()
      result = out.squeeze(0).squeeze(0).cpu().numpy()

      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.imshow(img, cmap='gray')
      plt.title('Original')

      plt.subplot(1, 2, 2)
      plt.imshow(result, cmap='gray')
      plt.title('Reconstructed')

      plt.show()


if __name__ == "__main__":
  # device = torch.device("cpu")
  device = torch.device("mps")
  print(f"[+] Using device: {device}")

  train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS)

  test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)
  test_loader = DataLoader(test_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS)
  inference_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=N_WORKERS)

  model = VQVAE(1, 1).to(device)
  train_losses, test_losses = [], []
  optim = torch.optim.Adam(model.parameters(), lr=1e-4)
  l2_loss = nn.MSELoss()

  out_path = "models/mnist_vqvae.pt"
  if LOAD:
    model.load_state_dict(torch.load(out_path))
    print("Loaded model from", out_path)
  else:
    for epoch in range(EPOCHS):
      try:
        print(f"[*] Epoch {epoch + 1}")
        train(model, train_loader, l2_loss, optim, train_losses)
        test(model, test_loader, l2_loss, test_losses)
      except KeyboardInterrupt:
        print("Training interrupted")
        break

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.show()

    torch.save(model.state_dict(), out_path)
    print("Model saved at", out_path)

  show_results(model, inference_loader)
