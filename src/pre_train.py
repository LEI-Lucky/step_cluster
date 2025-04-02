import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .modules import AutoEncoder
from .utils import Logger, save_checkpoint

def pretrain_autoencoder(model, train_loader, device, config, logger):
    # model => model.autoencoder
    """预训练自编码器"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    
    for epoch in range(config['training']['pretrain_epochs']):
        model.train()
        total_loss = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["pretrain_epochs"]}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            # 前向传播
            x_recon, _ = model(data)
            loss = criterion(x_recon, data)

            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # 只保存损失最小的一代和最后一代
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            save_checkpoint(model, optimizer, epoch, best_loss, 
                          config['paths']['save_dir'], 'pretrain_autoencoder_best')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                # 保存最后一代模型
                save_checkpoint(model, optimizer, epoch, avg_loss, 
                              config['paths']['save_dir'], 'pretrain_autoencoder_last')
                break
        
        # 如果是最后一个epoch，保存最后一代模型
        if epoch == config['training']['pretrain_epochs'] - 1:
            save_checkpoint(model, optimizer, epoch, avg_loss, 
                          config['paths']['save_dir'], 'pretrain_autoencoder_last')
    
    return model

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
    logger = Logger(config['paths']['log_dir'], 'pretrain')
    
    # 加载数据
    from .data_loader import load_mnist_data
    train_data, _ = load_mnist_data(config['paths']['data_dir'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['data']['train_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # 创建模型
    model = AutoEncoder(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        latent_dim=config['model']['latent_dim']
    )
    
    # 预训练模型
    model = pretrain_autoencoder(model, train_loader, device, config, logger)
    
    logger.info('预训练完成！')

if __name__ == '__main__':
    main() 