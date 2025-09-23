from src.models import StandardMLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import wandb
from tqdm import tqdm
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class TrainingArgs:
    model_path: str
    data_path: str
    epochs: int
    lr: float
    batch_size: int
    wandb_project: str

class MLPTrainer:
    def __init__(self, args: TrainingArgs):
        self.args = args

    def setup(self):
        self.examples_seen = 0

        normalise_and_flatten = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
        ])
        self.training_data = datasets.MNIST(
        root=self.args.data_path,
        train=True,
        download=True,
        transform=normalise_and_flatten,
        )

        self.test_data = datasets.MNIST(
        root=self.args.data_path,
        train=False,
        download=True,
        transform=normalise_and_flatten,
        )
    
        self.train_loader = DataLoader(self.training_data, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False)

        self.model = StandardMLP(in_features=28*28, out_features=10)

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.args.lr)

        wandb.init(project=self.args.wandb_project, config=self.args)
        wandb.watch(self.model, log="all", log_freq=1000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def train_step(self, images, labels):
        self.model.train()
        self.optimiser.zero_grad()

        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimiser.step()
        self.examples_seen += images.shape[0]
        wandb.log({"loss": loss.item()}, step=self.examples_seen)
        return loss

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.shape[0]
            accuracy = correct / total
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy


    def train(self):
        self.setup()
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader)
            for images, labels in pbar:
                loss = self.train_step(images, labels)
                pbar.set_postfix(loss=loss.item(), examples_seen=self.examples_seen)

            accuracy = self.evaluate()
            pbar.set_postfix(loss=loss.item(), examples_seen=self.examples_seen, accuracy=accuracy)

        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb_project", type=str)
    args = parser.parse_args()

    model_path = args.model_path

    args = TrainingArgs(**vars(args))
    trainer = MLPTrainer(args)
    trainer.train()

    #save model 
    torch.save(trainer.model.state_dict(), model_path)









