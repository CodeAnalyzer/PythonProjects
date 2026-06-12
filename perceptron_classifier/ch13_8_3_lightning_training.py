import pytorch_lightning as pl
import torch
import torch.nn as nn
import warnings
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

# Отключение предупреждений
warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()
        
        # Новые атрибуты PyTorch Lightning:
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        
        # Модель как в предыдущей секции:
        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers = [nn.Flatten()]
        
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
        
        all_layers.append(nn.Linear(hidden_units[-1], 10))
        self.model = nn.Sequential(*all_layers)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def prepare_data(self):
        MNIST(root=self.data_path, download=True)
    
    def setup(self, stage=None):
        mnist_all = MNIST(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )
        self.train, self.val = random_split(
            mnist_all, [55000, 5000], 
            generator=torch.Generator().manual_seed(1)
        )
        self.test = MNIST(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=0)


# ── Обучение модели ──
if __name__ == "__main__":
    torch.manual_seed(1)
    
    mnistclassifier = MultiLayerPerceptron()
    mnist_dm = MnistDataModule()
    
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices=1,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=50
        )
    else:
        trainer = pl.Trainer(
            max_epochs=10,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=50
        )
    
    trainer.fit(model=mnistclassifier, datamodule=mnist_dm)
    
    # ── Тестирование модели ──
    print('\nТестирование модели...')
    test_results = trainer.test(model=mnistclassifier, datamodule=mnist_dm)
    print(f'Результаты теста: {test_results}')
