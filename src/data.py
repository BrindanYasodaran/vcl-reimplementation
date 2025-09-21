import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms import Lambda
from torchvision.transforms import ToTensor
import argparse 



#code for generating permutations
def generate_permutations(seed, num_tasks, data_path):
    torch.manual_seed(seed)
    baseline = torch.arange(0, 28 * 28)
    permutations = [baseline] + [torch.randperm(28 * 28) for _ in range(num_tasks - 1)]

    #save permutations 
    torch.save(permutations, f"{data_path}/permutations.pt")

    return permutations


#code for dataset object 
class PermutedMNIST(Dataset):
    def __init__(self, base_dataset, permutation):
        self.base_dataset = base_dataset
        self.permutation = permutation
        self.transformation = ToTensor()

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.transformation(img)
        img = img.view(-1)
        img = img[self.permutation]
        return img, label

#code for loading the datasets across all times 
def load_datasets(base_dataset, seed, num_tasks, data_path):
    permutations = generate_permutations(seed, num_tasks, data_path)
    datasets = [PermutedMNIST(base_dataset, permutation) for permutation in permutations]
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    # parser.add_argument('--permutations_path', type=str)
    parser.add_argument('--num_tasks', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    training_data = datasets.MNIST(
        root=args.data_path,
        train=True,
        download=True,
    )

    test_data = datasets.MNIST(
        root=args.data_path,
        train=False,
        download=True,
    )

    train_datasets = load_datasets(training_data, args.seed, args.num_tasks, args.data_path)

    #check if original label is same as permuted 
    original_img, original_label = train_datasets[0][0]
    permuted_img_1, permuted_label_1 = train_datasets[1][0]
    permuted_img_2, permuted_label_2 = train_datasets[2][0]

    assert original_label == permuted_label_1
    assert not torch.equal(permuted_img_1, original_img)




