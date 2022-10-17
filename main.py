import torch
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.trainer import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys
import os
import uuid
from time import perf_counter


ROOT = "/home/captaintrojan/Projects/mnistest/results"

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

train, dev = torch.utils.data.random_split(mnist_trainset, [55000, 5000])
# train, dev, _ = torch.utils.data.random_split(mnist_trainset, [500, 500, 59000])

print(len(train), len(dev), len(test))


def make_loaders(batch_size):
    return {
        'train': torch.utils.data.DataLoader(train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4),

        'dev': torch.utils.data.DataLoader(dev,
                                           batch_size=100,
                                           shuffle=False,
                                           num_workers=4),

        'test': torch.utils.data.DataLoader(test,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=4),
    }


class MNISTSolver(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = torch.nn.Linear(28 * 28, 64)
        self.dense2 = torch.nn.Linear(64, 10)
        self.gelu = torch.nn.GELU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.dense1(x.view(x.size(0), -1))
        x = self.gelu(x)
        x = self.dense2(x)
        return self.softmax(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("loss/train", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("loss/eval", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("loss/eval_test", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def main(batch_size):
    start = perf_counter()
    model = MNISTSolver()
    loaders = make_loaders(batch_size)

    trainer = Trainer(accelerator='auto', max_epochs=1)
    trainer.fit(model, loaders['train'], loaders['dev'])
    test_loss_midway = trainer.test(model, loaders['test'])[0]['loss/eval_test']

    trainer = Trainer(accelerator='auto', callbacks=[EarlyStopping(monitor='loss/eval', mode='min')])
    trainer.fit(model, loaders['train'], loaders['dev'])
    test_loss_converged = trainer.test(model, loaders['test'])[0]['loss/eval_test']

    print("Results:")
    print(test_loss_midway, test_loss_converged)

    os.makedirs(os.path.join(ROOT, str(batch_size)), exist_ok=True)
    with open(os.path.join(ROOT, str(batch_size), str(uuid.uuid4())[:8] + ".txt"), "w") as f:
        f.write(f"{test_loss_midway},{test_loss_converged},{perf_counter() - start}\n")


if __name__ == '__main__':
    main(int(sys.argv[1]))
