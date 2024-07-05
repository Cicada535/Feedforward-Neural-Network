# Python 3.10.0

import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

print('')
print(f'Epoch\t\t Progress\t Loss\t\t Time')
print('')

# Определение набора данных
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Определение модели
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Загрузка данных
data = torch.randn(1000, 5)  # Пример случайных данных
labels = torch.randint(0, 10, (1000,))  # Пример случайных меток

# Создание объекта набора данных
dataset = MyDataset(data, labels)

# Создание загрузчика данных
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Создание модели
model = MyModel(input_size=5, output_size=10)

# Определение оптимизатора
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Определение функции потерь
criterion = nn.CrossEntropyLoss()

# Определение планировщика обучения с уменьшением шага
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# Обучение модели
epochs = 1000000001
patience = 10
best_loss = float('inf')
epoch_without_improvement = 0

logging.basicConfig(level=logging.INFO, filename='test.log',filemode='w', format="%(asctime)s %(levelname)s %(message)s")
logging.info('Training log:')
logging.info('----------------')

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямое распространение
        output = model(data)

        # Вычисление потерь
        loss = criterion(output, target)

        # Обратное распространение
        loss.backward()

        # Обновление весов
        optimizer.step()

        # Раннее прекращение 
        if loss.item() < best_loss:
            best_loss = loss.item()
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1
            if epoch_without_improvement >= patience:
                logging.basicConfig(level=logging.INFO, filename='test.log',filemode='w', format="%(asctime)s %(levelname)s %(message)s")
                logging.warning(f'Early termination of education at the epoch: {epoch}.')
                logging.warning('---------------------------------------------------')
                break  # Остановка обучения

        # Обновление шага обучения
        scheduler.step(loss)

        # Вывод прогресса
        if batch_idx % 10 == 0:
            t = time.strftime('%d/%m/%Y, %H:%M:%S', time.localtime())
            print(f'{epoch}\t\t [{batch_idx * len(data)}/{len(dataloader.dataset)}]\t {loss.item():.6f}    {t}')
            logging.basicConfig(level=logging.INFO, filename='test.log',filemode='w', format="%(asctime)s %(levelname)s %(message)s")
            logging.info('------------------------------------------')
            logging.info(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}]\t Loss: {loss.item():.6f}')
            logging.info('------------------------------------------')

# Сохранение модели
torch.save(model.state_dict(), 'model.pth')
