# 必要なモジュールを読み込む
# Flask関連
from flask import Flask, render_template, request, redirect, url_for, abort

# PyTorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

# Pillow(PIL)、datetime
from PIL import Image, ImageOps
from datetime import datetime

# モデルの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


device = torch.device("cpu")
model = 0
model = Net().to(device)
# 学習モデルをロードする
model.load_state_dict(
    torch.load("./mnist_cnn.pt", map_location=lambda storage, loc: storage)
)
model = model.eval()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)
        # 画像ファイルを読み込む
        image = Image.open(filepath)
        # PyTorchで扱えるように変換(リサイズ、白黒反転、正規化、次元追加)
        image = ImageOps.invert(image.convert("L")).resize((28, 28))
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        image = transform(image).unsqueeze(0)
        # 予測を実施
        output = model(image)
        _, prediction = torch.max(output, 1)
        result = prediction[0].item()

        return render_template("index.html", filepath=filepath, result=result)


if __name__ == "__main__":
    app.run(debug=True)
