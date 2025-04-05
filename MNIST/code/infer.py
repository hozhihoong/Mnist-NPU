from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from mindspore import Tensor, context, nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import io
import mindspore
import argparse
import os


app = Flask(__name__)

parser = argparse.ArgumentParser(description='MNIST Inference Server')
parser.add_argument('--model_name', type=str, default="lenet5_model.ckpt")
args = parser.parse_args()

context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)

# 定义网络结构
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load Model
def load_model(model_path):
    """加载模型检查点"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint {model_path} not found!")
    
    net = LeNet5().to_float(mindspore.float16)
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)
    return net

# 初始化模型
MODEL_DIR = "/home/mind/model"  # 模型存储目录 /home/mind/model
model_path = os.path.join(MODEL_DIR, args.model_name)  # 组合完整路径
print(f"Loading model from: {model_path}")
net = load_model(model_path)

# Preprocessing Function
def preprocess_image(image):
    img = image.convert('L').resize((32, 32))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 255.0 - 0.1307) / 0.3081
    return img_array.reshape(1, 1, 32, 32)

@app.route('/', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Debug: Print input shape
        print("Input shape to model:", processed_image.shape)
        
        output = net(Tensor(processed_image))
        predicted = int(np.argmax(output.asnumpy(), axis=1)[0])
        return jsonify({'prediction': predicted})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)