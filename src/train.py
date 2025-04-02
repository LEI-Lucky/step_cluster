import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from .modules import DeepClusteringModel
from .utils import Logger, save_checkpoint, save_experiment_results, load_checkpoint, plot_training_curves, log_best_results
from .metrics import calculate_metrics

def target_distribution(q):
    """计算目标分布"""
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_model(model, train_loader, test_loader, device, config, logger):
    """训练深度聚类模型"""
    # 加载预训练的自编码器权重
    pretrained_path = os.path.join(config['paths']['save_dir'], 'pretrain_autoencoder_best.pth')
    if os.path.exists(pretrained_path):
        logger.info(f"加载预训练权重: {pretrained_path}")
        checkpoint = load_checkpoint(pretrained_path)
        # 只加载自编码器部分的权重
        model.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        logger.info("预训练权重加载成功！")
    else:
        logger.warning(f"无法找到预训练权重: {pretrained_path}，将使用随机初始化的模型")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 初始化聚类中心
    logger.info("初始化聚类中心...")
    model.eval()
    features = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            _, z, _ = model(data)
            features.append(z.cpu().numpy())
    features = np.concatenate(features)
    kmeans = KMeans(n_clusters=config['model']['n_clusters'], random_state=config['seed'])
    kmeans.fit(features)
    model.clustering.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)  # 聚类中心
    
    # 训练循环
    best_loss = float('inf')
    best_metrics = None
    best_epoch = 0
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    update_interval = config['training']['update_interval']
    
    results = {
        'train_loss': [],
        'test_loss': [],
        'metrics': []
    }
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            x_recon, z, q = model(data)
            
            # 1️⃣ 计算重构损失
            recon_loss = nn.MSELoss()(x_recon, data)
            
            # 2️⃣ 计算聚类损失
            if batch_idx % update_interval == 0:
                q = target_distribution(q)
            
            # 3️⃣ 计算KL散度损失
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                q.log(), q.clone().detach().to(device)
            )
            
            # 4️⃣ 总损失
            loss = recon_loss + 0.1 * kl_loss
            
            # 5️⃣ 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 评估
        model.eval()
        test_loss = 0
        all_features = []
        all_labels = []
        all_q = []  # 收集所有批次的q值
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                x_recon, z, q = model(data)
                test_loss += nn.MSELoss()(x_recon, data).item()
                all_features.append(z.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_q.append(q.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        all_features = np.concatenate(all_features)
        all_labels = np.array(all_labels)
        all_q = np.concatenate(all_q)  # 合并所有批次的q值
        
        # 计算聚类标签
        pred_labels = np.argmax(all_q, axis=1)  # 使用合并后的all_q计算预测标签
        
        # 计算评估指标
        metrics = calculate_metrics(all_features, all_labels, pred_labels)
        
        # 记录结果
        results['train_loss'].append(avg_train_loss)
        results['test_loss'].append(avg_test_loss)
        results['metrics'].append(metrics)
        
        # 记录日志
        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f}')
        logger.info(f'  Test Loss: {avg_test_loss:.4f}')
        logger.info(f'  Metrics: \n {metrics}')
        
        # 早停检查
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_metrics = metrics
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            save_checkpoint(model, optimizer, epoch, best_loss, 
                          config['paths']['save_dir'], 'deep_clustering_best')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                # 保存最后一代模型
                save_checkpoint(model, optimizer, epoch, avg_test_loss, 
                              config['paths']['save_dir'], 'deep_clustering_last')
                break
        
        # 如果是最后一个epoch，保存最后一代模型
        if epoch == config['training']['epochs'] - 1:
            save_checkpoint(model, optimizer, epoch, avg_test_loss, 
                          config['paths']['save_dir'], 'deep_clustering_last')
    
    # 记录最佳结果
    log_best_results(best_epoch, best_loss, best_metrics, logger)
    
    # 绘制训练曲线
    curves_path = os.path.join(config['paths']['experiment_dir'], 'training_curves.png')
    plot_training_curves(results, curves_path, logger)
    
    return model, results

def main():
    """主函数"""
    # 加载配置
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # 初始化日志记录器
    logger = Logger(config['paths']['log_dir'], 'train')
    
    # 加载数据
    from .data_loader import load_mnist_data
    train_data, test_data = load_mnist_data(config['paths']['data_dir'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['data']['train_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['data']['test_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # 创建模型
    model = DeepClusteringModel(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=config['model']['latent_dim'],
        n_clusters=config['model']['n_clusters']
    )
    
    # 训练模型
    model, results = train_model(model, train_loader, test_loader, device, config, logger)
    
    # 保存实验结果
    save_experiment_results(results, config['paths']['experiment_dir'], 'deep_clustering')
    
    logger.info('训练完成！')

if __name__ == '__main__':
    main() 