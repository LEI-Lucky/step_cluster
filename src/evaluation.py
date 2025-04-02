import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from .metrics import calculate_metrics

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    all_features = []
    all_labels = []
    all_recon = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            x_recon, z, q = model(data)
            all_features.append(z.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_recon.append(x_recon.cpu().numpy())
    
    all_features = np.concatenate(all_features)
    all_labels = np.array(all_labels)
    all_recon = np.concatenate(all_recon)
    pred_labels = np.argmax(q.cpu().numpy(), axis=1)
    
    # 计算评估指标
    metrics = calculate_metrics(all_features, all_labels, pred_labels)
    
    return {
        'features': all_features,
        'labels': all_labels,
        'pred_labels': pred_labels,
        'recon': all_recon,
        'metrics': metrics
    }

def plot_tsne(features, labels, save_path):
    """绘制t-SNE可视化图"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Features')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(train_losses, test_losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics_history, save_path):
    """绘制评估指标变化曲线"""
    plt.figure(figsize=(12, 8))
    for metric_name in metrics_history[0].keys():
        values = [m[metric_name] for m in metrics_history]
        plt.plot(values, label=metric_name)
    
    plt.title('Evaluation Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def visualize_results(model, test_loader, device, save_dir):
    """可视化所有结果"""
    # 评估模型
    results = evaluate_model(model, test_loader, device)
    
    # 绘制t-SNE图
    plot_tsne(results['features'], results['labels'], 
              f'{save_dir}/tsne_true.png')
    plot_tsne(results['features'], results['pred_labels'], 
              f'{save_dir}/tsne_pred.png')
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['labels'], results['pred_labels'], 
                         f'{save_dir}/confusion_matrix.png')
    
    return results

def main():
    """主函数"""
    # 加载配置
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    from .data_loader import load_mnist_data
    _, test_data = load_mnist_data(config['paths']['data_dir'])
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['data']['test_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # 加载模型
    from .modules import DeepClusteringModel
    model = DeepClusteringModel(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=config['model']['latent_dim'],
        n_clusters=config['model']['n_clusters']
    )
    
    # 加载最佳模型权重
    import os
    checkpoint_path = os.path.join(config['paths']['save_dir'], 
                                 'deep_clustering_best.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 可视化结果
    results = visualize_results(model, test_loader, device, 
                              config['paths']['experiment_dir'])
    
    # 打印评估指标
    print("评估指标:")
    for metric_name, value in results['metrics'].items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main() 