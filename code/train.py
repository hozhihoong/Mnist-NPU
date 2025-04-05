import os
import argparse
import moxing as mox
import numpy as np
import mindspore
from mindspore import context, Tensor, Model, nn, save_checkpoint
from mindspore.common.initializer import Normal
from mindspore.train.callback import Callback, LossMonitor
from mindspore.dataset import vision, transforms
import mindspore.dataset as ds

# 环境初始化
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

class FinalCheckpoint(Callback):
    """自定义回调：仅在训练结束时保存模型"""
    def __init__(self, save_dir, prefix):
        super().__init__()
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(save_dir, exist_ok=True)

    def on_train_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        ckpt_path = os.path.join(
            self.save_dir, 
            f"{self.prefix}_model.ckpt"
        )
        save_checkpoint(cb_params.train_network, ckpt_path)
        print(f"Saved final model to {ckpt_path}")

class LeNet5(nn.Cell):
    """参数化的LeNet-5模型"""
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.in_channels, args.conv1_filters, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(args.conv1_filters, args.conv2_filters, 5, pad_mode='valid')
        self.fc1 = nn.Dense(args.conv2_filters*5*5, args.fc1_units, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(args.fc1_units, args.fc2_units, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(args.fc2_units, args.num_classes, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def create_dataset(data_dir, args, training=True):
    """创建可配置的数据管道"""
    dataset = ds.MnistDataset(
        os.path.join(data_dir, "train" if training else "test"),
        shuffle=training
    )
    
    # 图像预处理
    transforms_list = [
        vision.Resize((args.image_size, args.image_size)),
        vision.Rescale(1/255.0, 0.0),
        vision.Normalize(mean=[0.1307], std=[0.3081]),
        vision.HWC2CHW()
    ]
    
    # 标签处理
    type_cast_op = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(
        operations=transforms_list,
        input_columns="image",
        num_parallel_workers=args.num_workers
    )
    dataset = dataset.map(
        operations=type_cast_op,
        input_columns="label",
        num_parallel_workers=args.num_workers
    )
    
    return dataset.batch(
        args.batch_size if training else args.test_batch_size,
        drop_remainder=training
    )

def parse_args():
    """参数解析配置"""
    parser = argparse.ArgumentParser(description="MindSpore MNIST Training")
    
    # 路径参数
    parser.add_argument('--data_url', type=str, required=True)
    parser.add_argument('--train_url', type=str, required=True)
    
    # 模型参数
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--conv1_filters', type=int, default=6)
    parser.add_argument('--conv2_filters', type=int, default=16)
    parser.add_argument('--fc1_units', type=int, default=120)
    parser.add_argument('--fc2_units', type=int, default=84)
    parser.add_argument('--num_classes', type=int, default=10)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    
    # 系统参数
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100)
    
    return parser.parse_args()

def main():
    args = parse_args()
    local_data_dir = "./dataset"
    local_ckpt_dir = "./checkpoints"
    print(local_ckpt_dir)
    print(args.train_url)
    # 数据准备
    print("Downloading data...")
    mox.file.copy_parallel(args.data_url, local_data_dir)
    
    # 初始化组件
    net = LeNet5(args)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(
        params=net.trainable_params(),
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # 训练配置
    model = Model(net, loss, opt, metrics={"Accuracy": nn.Accuracy()})
    final_cb = FinalCheckpoint(local_ckpt_dir, "lenet5")
    loss_monitor = LossMonitor(args.log_interval)

    # 执行训练
    print("Start training...")
    train_ds = create_dataset(local_data_dir, args, training=True)
    model.train(
        args.epochs, 
        train_ds, 
        callbacks=[loss_monitor, final_cb],
        dataset_sink_mode=False
    )
    
    # 模型验证
    print("Start evaluation...")
    test_ds = create_dataset(local_data_dir, args, training=False)
    acc = model.eval(test_ds)
    print(f"Test Accuracy: {acc}")
    
    # 结果上传
    print("Uploading results...")
    mox.file.copy_parallel(local_ckpt_dir, args.train_url)
    print("uploading completed...")

if __name__ == "__main__":
    main()