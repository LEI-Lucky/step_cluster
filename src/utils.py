import os
import logging
import json
import torch
from datetime import datetime

class Logger:
    def __init__(self, log_dir, name):
        self.log_dir = log_dir
        self.name = name
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 设置日志文件
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器，指定编码为utf-8
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def info(self, message):
        self.logger.info(message)
        
    def error(self, message):
        self.logger.error(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def debug(self, message):
        self.logger.debug(message)

def save_experiment_results(results, save_dir, experiment_name):
    """保存实验结果"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(save_dir, f"{experiment_name}_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
def save_checkpoint(model, optimizer, epoch, loss, save_dir, name):
    """保存模型检查点"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # checkpoint_path = os.path.join(save_dir, f"{name}_epoch_{epoch}.pt")
    checkpoint_path = os.path.join(save_dir, f"{name}.pt")
    torch.save(checkpoint, checkpoint_path)
    
def load_checkpoint(checkpoint_path):
    """加载检查点并返回检查点对象"""
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_checkpoint_to_model(model, optimizer, checkpoint_path):
    """加载模型检查点到模型和优化器"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def plot_training_curves(results, save_path, logger=None):
    """绘制训练曲线并保存
    
    Args:
        results: 包含训练结果的字典，包括train_loss、test_loss和metrics
        save_path: 图像保存路径
        logger: 日志记录器
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 5))
        
        # 绘制训练和测试损失
        plt.subplot(1, 2, 1)
        epochs = range(1, len(results['train_loss']) + 1)
        plt.plot(epochs, results['train_loss'], 'b-', label='训练损失')
        plt.plot(epochs, results['test_loss'], 'r-', label='测试损失')
        plt.title('训练和测试损失')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制评估指标
        plt.subplot(1, 2, 2)
        metric_names = list(results['metrics'][0].keys())
        for metric_name in metric_names:
            metric_values = [metrics[metric_name] for metrics in results['metrics']]
            plt.plot(epochs, metric_values, label=metric_name)
        plt.title('评估指标')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        
        # 保存图像
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        if logger:
            logger.info(f"训练曲线已保存至: {save_path}")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"绘制训练曲线失败: {str(e)}")
        return False

def log_best_results(best_epoch, best_loss, best_metrics, logger):
    """记录最佳训练结果
    
    Args:
        best_epoch: 最佳轮次
        best_loss: 最佳损失值
        best_metrics: 最佳指标字典
        logger: 日志记录器
    """
    logger.info("=" * 50)
    logger.info(f"训练完成！最佳结果 (Epoch {best_epoch}):")
    logger.info(f"最佳测试损失: {best_loss:.4f}")
    logger.info("最佳评估指标:")
    for metric_name, value in best_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}") 