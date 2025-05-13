import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import tqdm  # 用于显示进度条
import re
from torchviz import make_dot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据清洗函数
def clean_text(text):
    # 确保 text 是字符串类型
    text = str(text)
    # 去除##之间的内容以及#
    text = re.sub(r'#.*?#', '', text)
    # 去除@提及部分
    text = re.sub(r'@\w+', '', text)
    # 去除超链接
    text = re.sub(r'http\S+', '', text)
    # 去除特殊字符，只保留中文
    text = re.findall(u'[\u4e00-\u9fa5]', str(text))
    # 去除多余空格
    text = ''.join(text).strip()
    return text

# 加载数据集
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        valid_labels = []
        for label in labels:
            if label in [0, 1]:  # 因为 weibo_senti_100k 是二分类，标签为 0 或 1
                valid_labels.append(label)
            else:
                print(f"Invalid label {label} found. Skipping this sample.")
        self.labels = valid_labels
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attn_scores = self.fc(lstm_output)
        attn_weights = self.softmax(attn_scores)
        weighted_sum = torch.sum(attn_weights * lstm_output, dim=1)
        return weighted_sum

# TextGCN 层
class TextGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TextGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, adj):
        support = torch.mm(inputs, self.weight)
        # 确保 adj 和 support 维度匹配
        assert adj.size(1) == support.size(0), f"adj dim 1 size {adj.size(1)} != support dim 0 size {support.size(0)}"
        output = torch.spmm(adj, support)
        return output

# 定义 BERT - TextGCN - BiLSTM 模型
class BERTTextGCNBiLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_size, num_classes, lstm_layers, dropout_rate=0.4):
        super(BERTTextGCNBiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.textgcn = TextGCNLayer(self.bert.config.hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_size)
        # 简化 DNN 层
        self.dnn = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, adj_base):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        batch_size, seq_len, _ = last_hidden_state.size()
        # 调整 last_hidden_state 维度
        last_hidden_state = last_hidden_state.view(-1, self.bert.config.hidden_size)
        # 构建 adj 矩阵
        identity_matrix = torch.eye(batch_size * seq_len).to(device)
        adj = identity_matrix
        textgcn_output = self.textgcn(last_hidden_state, adj)
        # 恢复 batch_size 和 seq_len 维度
        textgcn_output = textgcn_output.view(batch_size, seq_len, -1)
        lstm_output, _ = self.lstm(textgcn_output)
        lstm_output = self.dropout(lstm_output)
        attn_output = self.attention(lstm_output)
        dnn_output = self.dnn(attn_output)
        logits = self.fc(dnn_output)
        return logits

# 训练函数
def train(model, dataloader, criterion, optimizer, device, adj_base):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    # 显示训练数据加载进度条
    for batch in tqdm.tqdm(dataloader, desc="Training Batches"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, adj_base)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# 评估函数
def evaluate(model, dataloader, criterion, device, adj_base):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    # 显示评估数据加载进度条
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating Batches"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_ids, attention_mask, adj_base)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy, all_predicted, all_labels

# 预测函数
def predict(model, text, tokenizer, max_length, device, adj_base):
    model.eval()
    text = clean_text(text)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, adj_base)
        _, predicted = torch.max(logits.data, 1)
    return predicted.item()

# 加载数据
data = pd.read_csv('weibo_senti_100k.csv')
texts = data['review'].tolist()
labels = data['label'].tolist()

# 数据清洗
texts = [clean_text(text) for text in texts]

# 划分训练集、验证集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2,
                                                                      random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1,
                                                                    random_state=42)

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('/mnt/bert-base-chinese')
max_length = 128

# 创建数据集和数据加载器
train_dataset = WeiboDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = WeiboDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = WeiboDataset(test_texts, test_labels, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型
hidden_size = 512
lstm_layers = 3
dropout_rate = 0.4
model = BERTTextGCNBiLSTM('/mnt/bert-base-chinese', hidden_size, num_classes=2, lstm_layers=lstm_layers,
                          dropout_rate=dropout_rate).to(device)

# 定义损失函数和优化器，调整正则化强度
weight_decay = 1e-5
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=weight_decay)

# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# 构建基础邻接矩阵（这里基础邻接矩阵暂未使用）
bert_hidden_size = model.bert.config.hidden_size
adj_base = torch.randn(hidden_size, bert_hidden_size).to(device)

# 训练模型
num_epochs = 5
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch + 1}/{num_epochs}...")
    train_loss, train_accuracy = train(model, train_dataloader, criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
                                       device=device, adj_base=adj_base)
    val_loss, val_accuracy, _, _ = evaluate(model, val_dataloader, criterion=nn.CrossEntropyLoss(), device=device,
                                            adj_base=adj_base)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'checkpoint.pt')

    scheduler.step()

# 加载最佳模型
model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))

# 最终评估
test_loss, final_accuracy, all_predicted, all_labels = evaluate(model, test_dataloader, criterion=nn.CrossEntropyLoss(),
                                                                device=device, adj_base=adj_base)

# 计算其他评价指标
precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
recall = recall_score(all_labels, all_predicted, average='macro')
f1 = f1_score(all_labels, all_predicted, average='macro')

# 绘制训练结果图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_predicted)

# 打印最终结果
print(f'Final Accuracy: {final_accuracy:.4f}')
print(f'Final Precision: {precision:.4f}')
print(f'Final Recall: {recall:.4f}')
print(f'Final F1-score: {f1:.4f}')
print('Confusion Matrix:')
print(cm)

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 预测示例
test_texts_sample = test_texts[:5]
for text in test_texts_sample:
    prediction = predict(model, text, tokenizer, max_length, device, adj_base)
    print(f"Text: {text}")
    print(f"Predicted Label: {prediction}")
    print("-" * 50)
