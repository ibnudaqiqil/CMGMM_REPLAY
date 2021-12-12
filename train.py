import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST

from avalanche.training.strategies import Naive




class SimpleMLP(nn.Module):

    def __init__(self, num_classes=10, input_size=28*28):
        super(SimpleMLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(512, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x
# model
def main():

    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    perm_mnist = PermutedMNIST(n_experiences=3)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Continual learning strategy
    cl_strategy = Naive(
        model, optimizer, criterion, train_mb_size=32, train_epochs=2,
        eval_mb_size=32, device=device)

    # train and test loop
    results = []
    for train_task in train_stream:
        cl_strategy.train(train_task, num_workers=0)
        results.append(cl_strategy.eval(test_stream))


if __name__ == '__main__':
    main()
