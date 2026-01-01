import torch
from torch.utils.data import DataLoader
from dataset import PointCloudData, get_transforms
from model import PointNet, pointnetloss
import os

def main():
    # Настройки
    path = "ModelNet10"
    device = torch.device("cpu")
    print(f"Training on: {device}")
    
    # Датасеты
    train_ds = PointCloudData(path, transform=get_transforms(train=True))
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=get_transforms(train=False))
    
    # DataLoader
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")
    
    # Модель
    model = PointNet(classes=len(train_ds.classes))
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 4
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # [Batch, N, 3] -> [Batch, 3, N]
            inputs = data['pointcloud'].to(device).float().transpose(1, 2)
            labels = data['category'].to(device)
            
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs)
            
            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:    
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}], Loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        
        # Валидация
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs = data['pointcloud'].to(device).float().transpose(1, 2)
                labels = data['category'].to(device)
                outputs, __, __ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100. * correct / total
        print(f'Valid accuracy: {val_acc:.2f}%')
        
    # Сохраняем модель
    torch.save(model.state_dict(), "pointnet_model.pth")
    print("Model saved to pointnet_model.pth")

if __name__ == "__main__":
    main()