from src.models import StandardMLP
from src.data import load_datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
import wandb
from tqdm import tqdm
from dataclasses import dataclass
import os

@dataclass
class TrainingArgs:
    model_path: str
    data_path: str
    epochs: int
    lr: float
    batch_size: int
    wandb_project: str
    num_tasks: int
    seed: int


class NaiveMLPTrainer: 
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

        # --- MAJOR CHANGES HERE ---
        # 1. Use the StandardMLP, not the Bayesian one.
        self.model = StandardMLP(in_features=28*28, out_features=10)
        # 2. Do NOT load MLE weights. The baseline starts from random initialization.
        # 3. No coreset buffers are needed.

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # For a standard MLP, reduction='mean' is more conventional, but 'sum' is fine.
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        wandb.init(project=self.args.wandb_project, config=self.args, name="Naive-Baseline")
        wandb.watch(self.model, log="all", log_freq=1000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(self, images, labels):
        # --- SIMPLIFIED TRAINING STEP ---
        self.model.train()
        self.optimiser.zero_grad()
        
        images = images.to(self.device)
        labels = labels.to(self.device)

        # 1. Standard model returns only logits. No KL loss.
        logits = self.model(images)
        
        # 2. Loss is just CrossEntropy. No pi-weight scaling.
        loss = self.loss_fn(logits, labels)
        loss.backward()

        self.optimiser.step()
        self.examples_seen += images.shape[0]

        wandb.log({'loss': loss.item()}, step=self.examples_seen)
        return loss

    @torch.inference_mode()
    def evaluate(self, task_num):
        self.model.eval()
        accuracies = []

        for task in range(task_num + 1):
            correct = 0
            total = 0
            for images, labels in tqdm(self.test_loaders[task], desc=f"Evaluating Task {task}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # --- SIMPLIFIED EVALUATION ---
                # A standard model is deterministic. A single forward pass is enough.
                logits = self.model(images)
                pred = logits.argmax(dim=-1)

                correct += (pred == labels).sum().item()
                total += labels.shape[0]
            accuracy = correct / total
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)
        acc_dict = {"average_accuracy": avg_accuracy}
        for i, acc in enumerate(accuracies):
            acc_dict[f"task_{i}_accuracy"] = acc
        wandb.log(acc_dict, step=self.examples_seen)

        return avg_accuracy

    def train(self):
        # --- RADICALLY SIMPLIFIED MAIN LOOP ---
        self.setup()
        for task_num in range(self.args.num_tasks):
            # Phase 1: Train on the new task's data
            for epoch in range(self.args.epochs):
                self.model.train()
                pbar = tqdm(self.train_loaders[task_num], desc=f"Task {task_num} | Epoch {epoch+1}/{self.args.epochs}")
                for images, labels in pbar:
                    loss = self.train_step(images, labels)
                    pbar.set_postfix(loss=loss.item())

            # There is no coreset phase, state saving, prior updating, or state restoration.
            # We just evaluate after training is done for the task.
            accuracy = self.evaluate(task_num)
            print(f"Task {task_num} complete. Average Accuracy: {accuracy:.4f}")

            #save model
            file_path = f"{self.args.model_path}/task_{task_num}_mlp.pth"
            torch.save(self.model.state_dict(), file_path)

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Baseline for Permuted MNIST")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    # --- Use the same hyperparameters as your VCL run for a fair comparison ---
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    # --- REMOVED ARGUMENTS ---
    # parser.add_argument("--mle_weights_path", type=str)
    # parser.add_argument('--coreset_size', type=int)
    # parser.add_argument('--coreset_epochs', type=int)

    args = parser.parse_args()
    
    # Ensure the model save directory exists
    os.makedirs(args.model_path, exist_ok=True)
    
    # We can reuse the TrainingArgs dataclass, it will just ignore the extra fields
    # Or, for cleanliness, you could define a new NaiveTrainingArgs dataclass.
    training_args_dict = vars(args)
    
    args_dc = TrainingArgs(**training_args_dict)
    trainer = NaiveMLPTrainer(args_dc)
    trainer.train()