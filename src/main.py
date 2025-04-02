import argparse
import torch
import yaml
import os
from .pre_train import pretrain_autoencoder
from .train import train_model
from .evaluation import visualize_results
from .utils import Logger, log_best_results, plot_training_curves
from .modules import DeepClusteringModel
from .data_loader import load_mnist_data

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度聚类模型训练和评估')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--mode', type=str, default='all',
                      choices=['pretrain', 'train', 'evaluate', 'all'],
                      help='运行模式')
    parser.add_argument('--device', type=str, default=None,
                      help='运行设备 (cuda/cpu)')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # 初始化日志记录器
    logger = Logger(config['paths']['log_dir'], 'main')
    logger.info(f'使用设备: {device}')
    
    # 加载数据
    print("加载数据...")
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
    
    # 预训练阶段
    if args.mode in ['pretrain', 'all']:
        logger.info('开始预训练...')
        model.autoencoder = pretrain_autoencoder(
            model.autoencoder,
            train_loader,
            device,
            config,
            logger
        )
        logger.info('预训练完成！')
    
    # 训练阶段
    if args.mode in ['train', 'all']:
        logger.info('开始训练...')
        model, results = train_model(
            model,
            train_loader,
            test_loader,
            device,
            config,
            logger
        )
        logger.info('训练完成！')
        
        # 保存训练曲线
        curves_path = os.path.join(config['paths']['experiment_dir'], 'training_curves.png')
        plot_training_curves(results, curves_path, logger)
    
    # 评估阶段
    if args.mode in ['evaluate', 'all']:
        logger.info('开始评估...')
        results = visualize_results(
            model,
            test_loader,
            device,
            config['paths']['experiment_dir']
        )
        logger.info('评估完成！')
        
        # 打印评估指标
        logger.info('评估指标:')
        for metric_name, value in results['metrics'].items():
            logger.info(f'{metric_name}: {value:.4f}')

if __name__ == '__main__':
    main() 