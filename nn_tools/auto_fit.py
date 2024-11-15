from abc import abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
from typing import Protocol
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class History:
    def __init__(self):
        self.data = {}
    def append(self, data: dict[str, float]):
        for k, v in data.items():
            self.data[k] = self.data.get(k, []) + [v]
    def append_train(self, data: dict[str, float]):
        self.append({f"train_{k}": v for k, v in data.items()})
    def append_valid(self, data: dict[str, float]):
        self.append({f"valid_{k}": v for k, v in data.items()})
    def plot(self):
        df = pd.DataFrame(self.data)
        num_plots = len(set(k.split('_')[1] for k in df.columns))
        rows = cols = int(num_plots**0.5)
        if rows * cols < num_plots:
            cols += 1
        if rows * cols < num_plots:
            rows += 1

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
        axes = axes.flatten() if num_plots > 1 else [axes]
        
        for ax, col in zip(axes, set(k.split('_')[1] for k in df.columns)):
            df[[f"train_{col}", f"valid_{col}"]].plot(ax=ax)
            ax.set_title(col)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(col)
        
        for ax in axes[num_plots:]:
            fig.delaxes(ax)
        
        plt.tight_layout()

class AutoFit(Protocol):
    @abstractmethod
    def loss_weights(self) -> dict[str, float]:
        ...

    @abstractmethod
    def loss_fn(self, input_batch, output_batch) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def forward(self, input_batch):
        ...

    @abstractmethod
    def __call__(self, x):
        ...

    @staticmethod
    def current_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def relocate_batch(self, batch, device):
        if isinstance(batch, (tuple, list)):
            batch = tuple([b.to(device) if isinstance(b, torch.Tensor) else b for b in batch])
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        else:
            raise ValueError(f"Batch type {type(batch)} not supported")
        return batch

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, epochs: int, optimizer: torch.optim.Optimizer) -> History:
        device = self.current_device()
        history = History()
        for epoch in range(epochs):
            avg_losses = {}
            batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (train)")
            self.train()
            for bn, batch in enumerate(batches):
                batch = self.relocate_batch(batch, device)
                optimizer.zero_grad()
                out = self(batch)
                losses = self.loss_fn(batch, out)
                loss = sum(w * losses[k] for k, w in self.loss_weights().items())
                for k, v in losses.items():
                    avg_losses[k] = avg_losses.get(k, 0) + v.item()
                loss.backward()
                optimizer.step()
                batches.set_postfix({k: v / (bn + 1) for k, v in avg_losses.items()})
            history.append_train({k: v / (bn + 1) for k, v in avg_losses.items()})
            self.eval()
            avg_losses = {}
            with torch.no_grad():
                batches = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} (valid)")
                for bn, batch in enumerate(batches):
                    batch = self.relocate_batch(batch, device)
                    out = self(batch)
                    losses = self.loss_fn(batch, out)
                    for k, v in losses.items():
                        avg_losses[k] = avg_losses.get(k, 0) + v.item()
                    batches.set_postfix({k: v / (bn + 1) for k, v in avg_losses.items()})
            history.append_valid({k: v / (bn + 1) for k, v in avg_losses.items()})
        return history

    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str) -> "AutoFit":
        device = cls.current_device()
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model
        
                
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    class MLP(nn.Module, AutoFit):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 100),
                nn.LeakyReLU(),
                nn.Linear(100, 10)
            )
            self.loss = nn.CrossEntropyLoss()
        def forward(self, input_batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            x, _ = input_batch
            return self.model(x)
        def loss_weights(self) -> dict[str, float]:
            return {"ce": 1, "rnd": 0}
        def loss_fn(self, input_batch: tuple[torch.Tensor, torch.Tensor], output_batch: torch.Tensor) -> dict[str, torch.Tensor]:
            x, y_true = input_batch
            y_pred = output_batch
            return {"ce": self.loss(y_pred, y_true), "rnd": torch.rand(1)}
        
    mlp = MLP()
    optim = torch.optim.Adam(mlp.parameters(), lr=0.001)
    h = mlp.fit(trainloader, testloader, 10, optim)
    h.plot()
    plt.show()

