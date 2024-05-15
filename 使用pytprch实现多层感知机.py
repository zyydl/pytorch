import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 准备数据集
input_size = 100
output_size = 5

# 生成训练数据和标签
X_train = torch.randn(1000, input_size)  # 1000个训练样本
y_train = torch.randint(0, output_size, (1000,))  # 随机生成标签，范围为0到output_size-1

# 生成测试数据和标签
X_test = torch.randn(200, input_size)  # 200个测试样本
y_test = torch.randint(0, output_size, (200,))  # 随机生成标签，范围为0到output_size-1

# 将数据转换为TensorDataset和DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = MLP(input_size, hidden_size=64, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy on test set: {accuracy:.2f}')

# 可视化部分测试集样本及其预测结果
num_samples = 5
print("Sample\t\tPredicted\tTrue Label")
for i in range(num_samples):
    print(f"{i + 1}:\t\t{predicted[i].item()}\t\t{y_test[i].item()}")

# 可以根据需要进行其他可视化或分析
