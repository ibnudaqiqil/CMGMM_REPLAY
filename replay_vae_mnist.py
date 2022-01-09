from torch.utils.data import DataLoader
import sys
#sys.path.append('C:\\Anaconda3\\envs\\pytorch\\lib\\site-packages')
from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import split_train_val

dataset = MNIST("my/data/path", download=True, train=True)
scenario = ClassIncremental(
    dataset,
    increment=1,
    initial_increment=5
)

print(f"Number of classes: {scenario.nb_classes}.")
print(f"Number of tasks: {scenario.nb_tasks}.")

for task_id, train_taskset in enumerate(scenario):
    train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = DataLoader(train_taskset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_taskset, batch_size=32, shuffle=True)
