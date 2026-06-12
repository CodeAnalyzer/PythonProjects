import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def prepare_data(self):
        MNIST(root=self.data_path, download=True)
    
    def setup(self, stage=None):
        # stage может принимать значения
        # 'fit', 'validate', 'test', or 'predict'
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


# ── Инициализация модуля данных ──
if __name__ == "__main__":
    torch.manual_seed(1)
    mnist_dm = MnistDataModule()
    mnist_dm.prepare_data()
    mnist_dm.setup()
    
    print('Размеры наборов данных:')
    print(f'  Обучение: {len(mnist_dm.train)}')
    print(f'  Валидация: {len(mnist_dm.val)}')
    print(f'  Тест: {len(mnist_dm.test)}')
