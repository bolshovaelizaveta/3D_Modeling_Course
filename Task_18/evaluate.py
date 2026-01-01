import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import itertools
import os
from torch.utils.data import DataLoader
from dataset import PointCloudData, get_transforms
from model import PointNet

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    print("Confusion Matrix saved to results/confusion_matrix.png")

def visualize_sample(points, true_label, pred_label):
    points = points.numpy()
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(size=2, color='royalblue')
    )])
    fig.update_layout(
        title=f"True: {true_label}, Predicted: {pred_label}",
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.show()

def main():
    device = torch.device("cpu")
    path = "ModelNet10"
    
    # Датасет
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=get_transforms(train=False))
    valid_loader = DataLoader(dataset=valid_ds, batch_size=32, shuffle=False, num_workers=0)
    
    classes_dict = {v: k for k, v in valid_ds.classes.items()}
    
    # Модель
    model = PointNet(classes=len(classes_dict))
    model.load_state_dict(torch.load("pointnet_model.pth", map_location=device)) # Указываем map_location для CPU
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # CPU
            inputs = data['pointcloud'].to(device).float().transpose(1, 2)
            labels = data['category'].to(device)
            
            outputs, __, __ = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.cpu().numpy()) 
            all_labels += list(labels.cpu().numpy())
            
    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [classes_dict[i] for i in sorted(classes_dict.keys())] 
    plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')
    
    # Визуализация одного рандомного семпла
    idx_to_visualize = np.random.randint(0, len(valid_ds)) 
    sample = valid_ds[idx_to_visualize]
    pcd = sample['pointcloud']
    true_cls_idx = sample['category']
    true_cls_name = classes_dict[true_cls_idx]
    
    # Прогон через модель
    inp = pcd.unsqueeze(0).transpose(1, 2).to(device)
    out, _, _ = model(inp)
    pred_cls_idx = out.argmax(dim=1).item()
    pred_cls_name = classes_dict[pred_cls_idx]
    
    print(f"Visualizing random sample. True: {true_cls_name}, Predicted: {pred_cls_name}")
    visualize_sample(pcd, true_cls_name, pred_cls_name)

if __name__ == "__main__":
    main()