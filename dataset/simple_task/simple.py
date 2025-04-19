# ✅ 通用 MLP 训练器 + 3 个任务示例（文本分类 + 图像分类）
import os
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from util.model import print_model_param_count, set_seed
from datasets import load_dataset
# ✅ 通用 MLP 模型


mnist_model = './mnist_model.pt'

fashion_mnist_model = './fashion_mnist_model.pt'

text_classification_model = './text_classification_model.pt'

imdb_classification_model = './imdb_classification_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def create_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    """
    构建多层 MLP。

    参数：
    - input_dim (int): 输入特征维度
    - hidden_dims (list[int]): 每层隐藏层的维度列表，例如 [128, 64]
    - output_dim (int): 输出维度
    - activation (nn.Module): 激活函数类（默认是 ReLU）

    返回：
    - nn.Sequential 模型
    """

    # ✅ 参数合法性检查
    assert isinstance(input_dim, int) and input_dim > 0, "input_dim must be a positive integer"
    assert isinstance(output_dim, int) and output_dim > 0, "output_dim must be a positive integer"
    assert isinstance(hidden_dims, list) and len(hidden_dims) > 0, "hidden_dims must be a non-empty list"
    assert all(isinstance(h, int) and h > 0 for h in hidden_dims), "All hidden layer sizes must be positive integers"
    assert callable(activation), "activation must be a callable nn.Module class (e.g., nn.ReLU)"

    # ✅ 构造网络层
    layers = []
    dims = [input_dim] + hidden_dims

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], output_dim))  # 输出层（无激活）

    return nn.Sequential(*layers)

def train_model(model, trainloader, optimizer, criterion, epochs=100, validloader=None, early_stop_patience=15):
    best_val_acc = 0.0
    best_model_state = None
    no_improve_count = 0  # 连续未提升计数器

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Acc = {acc:.4f}")

        # 验证阶段
        if validloader:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x_val, y_val in validloader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    output_val = model(x_val)
                    loss_val = criterion(output_val, y_val)
                    val_loss += loss_val.item()
                    val_pred = output_val.argmax(dim=1)
                    val_correct += (val_pred == y_val).sum().item()
                    val_total += y_val.size(0)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # 判断是否有提升
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                no_improve_count = 0
                print(f"✅ New best model with Val Acc = {best_val_acc:.4f}")
            else:
                no_improve_count += 1
                print(f"⏸️ No improvement for {no_improve_count} epoch(s)")

            # 提前停止判断
            if no_improve_count >= early_stop_patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1} (no improvement in {early_stop_patience} epochs)")
                break

            model.train()

    # 恢复最佳模型参数
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with Val Acc = {best_val_acc:.4f}")

# ✅ 图像分类任务：MNIST

def run_mnist(model):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n🖼️ Training on MNIST")
    train_model(model, trainloader, optimizer, criterion, epochs=100, validloader=valloader)

# ✅ 图像分类任务：Fashion-MNIST（含 validation）
def run_fashion_mnist(model):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n🧥 Training on Fashion-MNIST")
    train_model(model, trainloader, optimizer, criterion, epochs=100, validloader=valloader)

# ✅ 文本分类任务：20 Newsgroups (binary version)
def run_text_classification(model):
    categories = ['sci.med', 'rec.sport.hockey']
    data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data.data).toarray()
    y = np.array(data.target)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    trainloader = DataLoader(list(zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))), batch_size=64, shuffle=True)
    valloader = DataLoader(list(zip(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))), batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n📝 Training on 20Newsgroups (sci.med vs hockey)")
    train_model(model, trainloader, optimizer, criterion, epochs=100, validloader=valloader)


def run_trec_classification(model):
    print("\n❓ Training on TREC (Question Type Classification)")

    dataset = load_dataset("trec")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["coarse_label"]
    val_texts = dataset["test"]["text"]
    val_labels = dataset["test"]["coarse_label"]

    vectorizer = TfidfVectorizer(max_features=2000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    trainloader = DataLoader(list(zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))), batch_size=64, shuffle=True)
    valloader = DataLoader(list(zip(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))), batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    train_model(model, trainloader, optimizer, criterion, epochs=100, validloader=valloader)


def run_imdb_classification(model):
    print("\n🎬 Training on IMDb (Sentiment Classification)")

    # ✅ 加载 IMDb 数据（正负影评）
    dataset = load_dataset("imdb")


    # ✅ 可选采样（加速实验）
    train_data = dataset["train"]

    val_data = dataset["test"]

    # ✅ 文本与标签
    train_texts = train_data["text"]
    train_labels = train_data["label"]
    val_texts = val_data["text"]
    val_labels = val_data["label"]

    # ✅ TF-IDF 提取
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # ✅ 转 Tensor + DataLoader
    trainloader = DataLoader(list(zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))), batch_size=64, shuffle=True)
    valloader = DataLoader(list(zip(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))), batch_size=64)

    # ✅ 模型训练
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    train_model(model, trainloader, optimizer, criterion, epochs=100, validloader=valloader)


def get_model_structure(model_name):
    if model_name == 'mnist':
        return create_mlp(input_dim=784, hidden_dims=[1024, 768, 512, 384, 256, 128, 64, 32], output_dim=10, activation=nn.ReLU())
    elif model_name == 'fashion_mnist':
        return create_mlp(input_dim=784, hidden_dims=[1024, 768, 512, 384, 256, 128, 64, 32], output_dim=10, activation=nn.ReLU())
    elif model_name == 'text_classification':
        return create_mlp(input_dim=1000, hidden_dims=[1024, 768, 512, 384, 256, 128, 64, 32], output_dim=2, activation=nn.ReLU())
    
    elif model_name == 'imdb_classification':
        return create_mlp(input_dim=2000, hidden_dims=[1024, 768, 512, 384, 256, 128, 64, 32], output_dim=2, activation=nn.ReLU())
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_mnist_model(path=mnist_model):
    model = get_model_structure('mnist')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model

def get_fashion_mnist_model(path=fashion_mnist_model):
    model = get_model_structure('fashion_mnist')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model

def get_text_classification_model(path=text_classification_model):
    model = get_model_structure('text_classification')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model

def get_imdb_classification_model(path=imdb_classification_model):
    model = get_model_structure('imdb_classification')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model

# ✅ 启动任务
def main(
    seed=42,
    dir_name='mnist',
):


    set_seed(seed=seed)

    os.makedirs(dir_name, exist_ok=True)



    ###  0.9842 create_mlp(input_dim=784, hidden_dims=[1024, 512, 256, 128, 64, 32], output_dim=10, activation=nn.ReLU()) 1,503,530

    model = get_model_structure('mnist').to(device)
    
    run_mnist(model=model)

    print_model_param_count(model)

    torch.save(model.state_dict(), dir_name + '/' +  mnist_model)

    ###  0.9051 create_mlp(input_dim=784, hidden_dims=[2048, 1024, 512, 256, 128, 64, 32], output_dim=10, activation=nn.ReLU())

    model = get_model_structure('fashion_mnist').to(device)

    run_fashion_mnist(model=model)

    torch.save(model.state_dict(), dir_name + '/' + fashion_mnist_model)

    print_model_param_count(model)

    ## 0.9540 create_mlp(input_dim=1000, hidden_dims=[2048, 1024, 512, 256, 128, 64, 32], output_dim=2, activation=nn.ReLU())

    model = get_model_structure('text_classification').to(device)

    run_text_classification(model=model)


    torch.save(model.state_dict(), dir_name + '/' + text_classification_model)

    print_model_param_count(model)


    ## ✅ Loaded best model with Val Acc =  0.8780

    ## [4096, 2048, 1024, 768, 512, 384, 256, 128, 64, 32]


    model = get_model_structure('imdb_classification').to(device)

    run_imdb_classification(model=model)

    torch.save(model.state_dict(), dir_name + '/' + imdb_classification_model)

    print_model_param_count(model)

    # 0.8340

    # model = get_model_structure('sst2_classification').to(device)

    # run_trec_classification(model=model)

    # torch.save(model.state_dict(), sst2_classification_model)

    # print_model_param_count(model)
