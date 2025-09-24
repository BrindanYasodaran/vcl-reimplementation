from src.models import StandardMLP, BayesianMLP
from src.utils import load_mle_weights
from src.data import load_datasets
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
import os

@dataclass
class TrainingArgs:
    model_path: str
    data_path: str
    mle_weights_path: str
    epochs: int
    lr: float
    batch_size: int
    wandb_project: str
    num_tasks: int
    seed: int
    coreset_size: int
    coreset_epochs: int

class MLPTrainer:
    def __init__(self, args: TrainingArgs):
        self.args = args

    def setup(self):
        self.examples_seen = 0

        base_training_data = datasets.MNIST(
        root=self.args.data_path,
        train=True,
        download=True,
        )

        base_test_data = datasets.MNIST(
        root=self.args.data_path,
        train=False,
        download=True,
        )

        self.training_data = load_datasets(base_training_data, self.args.seed, self.args.num_tasks, self.args.data_path)
        self.test_data = load_datasets(base_test_data, self.args.seed, self.args.num_tasks, self.args.data_path)

        self.num_train_data = len(self.training_data[0])

        self.train_loaders = [DataLoader(training_data, batch_size=self.args.batch_size, shuffle=True) for training_data in self.training_data]
        self.test_loaders = [DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False) for test_data in self.test_data]

        self.model = BayesianMLP(in_features=28*28, out_features=10)
        load_mle_weights(self.model, self.args.mle_weights_path)

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        wandb.init(project=self.args.wandb_project, config=self.args)
        wandb.watch(self.model, log="all", log_freq=1000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.coreset_imgs = []
        self.coreset_labels = []


    def train_step(self, images, labels, coreset=False) :
        self.model.train()
        self.optimiser.zero_grad()
        
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits, kl_loss = self.model(images)
        
        nll_loss = self.loss_fn(logits, labels)

        if coreset:
            pi_weight = len(self.coreset_imgs) / len(images)
        else:
            pi_weight = self.num_train_data / len(images)

        loss = (pi_weight * nll_loss) + kl_loss
        loss.backward()

        self.optimiser.step()

        if not coreset:
            self.examples_seen += images.shape[0]

        #if nll_loss is much smaller than kl_loss then scale nll_loss by len(dataset)/len(batch)
        # wandb.log({f'task_{task_num}_loss': loss.item(), f'task_{task_num}_nll_loss': nll_loss.item(), f'task_{task_num}_kl_loss': kl_loss.item()}, step=self.examples_seen)
        if not coreset:
            wandb.log({'loss': loss.item(), 'nll_loss': nll_loss.item(), 'kl_loss': kl_loss.item()}, step=self.examples_seen)
        return loss

    @torch.inference_mode()
    def evaluate(self, task_num):
        self.model.eval()
        accuracies = []

        for task in range(task_num + 1):
            correct = 0
            total = 0
            for images, labels in tqdm(self.test_loaders[task]):
                images = images.to(self.device)
                labels = labels.to(self.device)

                avg_log_probs = self.model.predict(images, num_samples=5)
                pred = avg_log_probs.argmax(dim=-1)

                correct += (pred == labels).sum().item()
                total += labels.shape[0]
            accuracy = correct / total
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)
        acc_dict = {"average_accuracy" : avg_accuracy}
        for i, acc in enumerate(accuracies):
            acc_dict[f"task_{i}_accuracy"] = acc
        wandb.log(acc_dict, step=self.examples_seen)

        return avg_accuracy

    def update_coreset(self, task_num):
        #add random subset of training data to coreset 
        current_dataset = self.training_data[task_num]
        random_indices = torch.randperm(len(current_dataset))[:self.args.coreset_size]
        subset = torch.utils.data.Subset(current_dataset, random_indices)

        #slow loop (not sure if this can be vectorised)
        for img_tensor, label in subset:
            self.coreset_imgs.append(img_tensor)
            self.coreset_labels.append(label)

    def train_on_coreset(self):
        coreset_dataset = torch.utils.data.TensorDataset(torch.stack(self.coreset_imgs), torch.LongTensor(self.coreset_labels))
        coreset_loader = DataLoader(coreset_dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.coreset_epochs):
            self.model.train()
            pbar = tqdm(coreset_loader)
            for images, labels in pbar:
                loss = self.train_step(images, labels, coreset=True)


    def train(self):
        self.setup()
        for task_num in range(self.args.num_tasks):
            # self.examples_seen = 0
            # accuracy = self.evaluate(task_num)

            for epoch in range(self.args.epochs):
                self.model.train()
                pbar = tqdm(self.train_loaders[task_num])
                for images, labels in pbar:
                    loss = self.train_step(images, labels)
                    pbar.set_postfix(task_num=task_num, loss=loss.item(), examples_seen=self.examples_seen)

            wandb.log({'task_boundary': task_num}, step=self.examples_seen)

            #save model 
            file_path = f"{self.args.model_path}/task_{task_num}_mlp.pth"
            torch.save(self.model.state_dict(), file_path)
            self.model.update_prior()

            if self.args.coreset_size > 0:
                self.update_coreset(task_num)
                self.train_on_coreset()

            accuracy = self.evaluate(task_num)
            print(f"Task {task_num} complete. Average Accuracy: {accuracy:.4f}")

            #load model params before fine tuning on coreset 
            self.model.load_state_dict(torch.load(file_path))
            self.model.update_prior()

        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mle_weights_path", type=str)
    parser.add_argument('--coreset_size', type=int, default=200)
    parser.add_argument('--coreset_epochs', type=int, default=5)

    args = parser.parse_args()

    args = TrainingArgs(**vars(args))
    trainer = MLPTrainer(args)
    trainer.train()

