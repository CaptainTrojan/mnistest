from asyncio import base_tasks
import torch
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.trainer import Trainer
import pytorch_lightning as pl


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

train, dev = torch.utils.data.random_split(mnist_trainset, [55000, 5000])

print(len(train), len(dev), len(test))

def make_loaders(batch_size):
    return {
        'train' : torch.utils.data.DataLoader(train, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=4),

        'dev' : torch.utils.data.DataLoader(dev, 
                                            batch_size=100, 
                                            shuffle=False, 
                                            num_workers=4),
        
        'test'  : torch.utils.data.DataLoader(test, 
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
        self.softmax = torch.nn.Softmax(dim=-1)
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-3)

model = MNISTSolver()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
lossfn = torch.nn.CrossEntropyLoss()
trainer = Trainer(accelerator='gpu')
loaders = make_loaders(16)

trainer.fit(model, loaders['train'], loaders['dev'])
