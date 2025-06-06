import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertModel, BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import math
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据准备
class WeiboDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, slang_mapping):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.slang_mapping = slang_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['review']  # 修改为 'review'
        sentiment = self.data.iloc[index]['label']

        # 应用文本清洗
        text = clean_text(text, self.slang_mapping)

        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded_text['input_ids'].flatten()
        attention_mask = encoded_text['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }

# 2. 自定义 BiLSTM 单元
class SentimentAwareLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentimentAwareLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_i2h = nn.Linear(input_size, 4 * hidden_size)
        self.linear_h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hidden_state, sentiment_intensity):
        h_prev, c_prev = hidden_state

        # 计算门控值
        lstm_input = self.linear_i2h(input) + self.linear_h2h(h_prev)
        i, f, g, o = torch.split(lstm_input, self.hidden_size, dim=-1)

        # 调整门控机制
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        # 新遗忘门 = 原遗忘门 * (1 + sentiment_intensity)
        f = f * (1 + sentiment_intensity)
        # 新输入门 = 原输入门 * 情感强度值
        i = i * sentiment_intensity

        # 计算 cell state 和 hidden state
        c = f * c_prev + i * torch.tanh(g)
        h = o * torch.tanh(c)

        return h, c

# 3. 自定义 BiLSTM
class SentimentAwareBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentimentAwareBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell_forward = SentimentAwareLSTMCell(input_size, hidden_size)
        self.lstm_cell_backward = SentimentAwareLSTMCell(input_size, hidden_size)

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return (h_0, c_0)

    def forward(self, input, sentiment_intensity):
        batch_size, seq_len, input_size = input.size()
        device = input.device

        # 初始化 hidden state
        hidden_state_forward = self.init_hidden(batch_size, device)
        hidden_state_backward = self.init_hidden(batch_size, device)

        # 前向计算
        output_forward = []
        for t in range(seq_len):
            input_t = input[:, t, :]
            hidden_state_forward = self.lstm_cell_forward(input_t, hidden_state_forward, sentiment_intensity[:, t].unsqueeze(1))
            output_forward.append(hidden_state_forward[0])
        output_forward = torch.stack(output_forward, dim=1)

        # 后向计算
        output_backward = []
        for t in reversed(range(seq_len)):
            input_t = input[:, t, :]
            hidden_state_backward = self.lstm_cell_backward(input_t, hidden_state_backward, sentiment_intensity[:, t].unsqueeze(1))
            output_backward.append(hidden_state_backward[0])
        output_backward = torch.stack(output_backward, dim=1)
        output_backward = torch.flip(output_backward, dims=[1])

        # 合并前向和后向输出
        output = torch.cat((output_forward, output_backward), dim=2)
        return output

# 4. 注意力机制模块
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # Wa
        self.gamma = nn.Parameter(torch.randn(1))  # gamma
        self.hidden_dim = hidden_dim * 2

    def forward(self, bilstm_output, sentiment_intensity):
        batch_size, seq_len, hidden_dim = bilstm_output.size()
        # 初始化 s_prev
        s_prev = torch.zeros(batch_size, hidden_dim).to(bilstm_output.device)

        # 计算注意力分数
        attention_scores = []
        for t in range(seq_len):
            h_t = bilstm_output[:, t, :]  # (batch_size, hidden_dim)
            gamma_t = sentiment_intensity[:, t].unsqueeze(1)  # (batch_size, 1)

            # 计算注意力分数
            score = torch.sum((torch.tanh(self.Wa(h_t)) + gamma_t) * s_prev, dim=1)  # (batch_size,)
            attention_scores.append(score)

            # 更新 s_prev
            s_prev = h_t

        # 将注意力分数堆叠起来
        attention_scores = torch.stack(attention_scores, dim=1)  # (batch_size, seq_len)

        # 使用 Softmax 函数进行归一化处理
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # 加权求和
        attended_output = torch.sum(bilstm_output * attention_weights.unsqueeze(2), dim=1)  # (batch_size, hidden_dim)

        return attended_output

# 5. 自定义 DNN 层
class AdaptiveDNN(nn.Module):
    def __init__(self, input_dim, num_classes, class_weights=None):
        super(AdaptiveDNN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.d_min = 256
        self.d_max = 1024

        # 动态计算隐藏层维度
        self.hidden_dim = self.d_min  # 初始值，会在 forward 中更新

        # 定义线性层
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, class_weights=None):
        # 计算输入特征方差
        feature_var = torch.var(x, dim=1)  # 计算每个样本的特征方差

        # 动态确定隐藏层维度
        self.hidden_dim = self.d_min + int(feature_var.mean().item() * (self.d_max - self.d_min))
        self.hidden_dim = max(self.d_min, min(self.hidden_dim, self.d_max))  # 限制在 d_min 和 d_max 之间

        # 重新定义线性层 (如果 hidden_dim 改变)
        if self.linear1.out_features != self.hidden_dim:
            self.linear1 = nn.Linear(self.input_dim, self.hidden_dim).to(x.device)
            self.linear2 = nn.Linear(self.hidden_dim, self.num_classes).to(x.device)

        # 前向传播
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        # 加权 Softmax
        if class_weights is not None:
            x = x * class_weights

        return x

# 6. 模型定义
class BERT_TextGCN_BiLSTM(nn.Module):
    def __init__(self, bert_model_name, gcn_hidden_dim, lstm_hidden_dim, num_classes, vocab_size, embedding_dim,
                 sentiment_conv_filters, sentiment_fc_dim, class_weights=None):
        super(BERT_TextGCN_BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.gcn1 = GCNConv(embedding_dim, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.input_linear = nn.Linear(embedding_dim, gcn_hidden_dim)  # 添加线性层

        # 情感强度预测子网络
        self.sentiment_conv1 = nn.Conv1d(embedding_dim, sentiment_conv_filters, kernel_size=3, padding=1)
        self.sentiment_conv2 = nn.Conv1d(sentiment_conv_filters, sentiment_conv_filters, kernel_size=3, padding=1)
        self.sentiment_fc = nn.Linear(sentiment_conv_filters * gcn_hidden_dim, sentiment_fc_dim)
        self.sentiment_output = nn.Linear(sentiment_fc_dim, 1)  # 输出情感强度值

        # 使用自定义 BiLSTM
        self.bilstm = SentimentAwareBiLSTM(gcn_hidden_dim, lstm_hidden_dim)
        # 注意力机制
        self.attention = Attention(lstm_hidden_dim)
        # 自定义 DNN 层
        self.dnn = AdaptiveDNN(lstm_hidden_dim * 2, num_classes, class_weights)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_weights = class_weights

        # 定义模型参数
        self.W_GCN = nn.ParameterList([param for param in self.gcn1.parameters()])
        self.W_GCN.extend([param for param in self.gcn2.parameters()])
        self.W_BiLSTM = nn.ParameterList([param for param in self.bilstm.parameters()])

        # 初始化动态图的邻接矩阵
        self.adj_matrix = None

    def forward(self, input_ids, attention_mask, edge_index):
        # BERT Embedding
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.last_hidden_state

        # TextGCN
        x = self.embedding(input_ids)  # input_ids 作为节点特征
        # 残差连接
        x_initial = self.input_linear(x)  # 将输入特征映射到 GCN 隐藏层维度
        x = F.relu(self.gcn1(x, edge_index))
        x = x + x_initial  # 残差连接
        x = F.relu(self.gcn2(x, edge_index))

        # 情感强度预测
        sentiment_input = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        sentiment_output = F.relu(self.sentiment_conv1(sentiment_input))
        sentiment_output = F.relu(self.sentiment_conv2(sentiment_output))
        sentiment_output = sentiment_output.view(sentiment_output.size(0), -1)  # Flatten
        sentiment_output = F.relu(self.sentiment_fc(sentiment_output))
        sentiment_intensity = self.sentiment_output(sentiment_output)  # 情感强度值

        # 归一化情感强度值
        sentiment_intensity = torch.sigmoid(sentiment_intensity)  # 使用 sigmoid 函数将情感强度值映射到 0 到 1 之间

        # BiLSTM
        bilstm_input = x
        bilstm_output = self.bilstm(bilstm_input, sentiment_intensity)

        # 注意力机制
        attended_output = self.attention(bilstm_output, sentiment_intensity)

        # DNN
        output = self.dnn(attended_output, self.class_weights)
        return output

    def compute_dynamic_graph(self, input_ids):
        # 1. 计算动态余弦相似度
        # 获取所有词的embedding
        word_embeddings = self.embedding(torch.arange(self.vocab_size).to(input_ids.device))  # (vocab_size, embedding_dim)

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(word_embeddings.detach().cpu().numpy())  # (vocab_size, vocab_size)
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float)

        # 2. 构建动态图邻接矩阵
        # 设置阈值，只保留相似度高于阈值的边
        threshold = 0.5
        adj_matrix = (similarity_matrix > threshold).int()

        # 转换为边的索引
        edge_index = torch.nonzero(adj_matrix).t().contiguous()

        # 保存当前的邻接矩阵
        self.prev_adj_matrix = self.adj_matrix
        self.adj_matrix = adj_matrix

        return edge_index

# 7. 训练代码
def train(model, data_loader, optimizer, device, build_graph, tokenizer, class_weights, lambda_1, lambda_2, lambda_3):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment = batch['sentiment'].to(device)

        # 计算动态图
        corpus = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        edge_index, _ = build_graph(corpus, tokenizer)  # 获取 edge_index，忽略 word_index
        edge_index = edge_index.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, edge_index)

        # 1. 加权交叉熵损失
        L_bal = F.cross_entropy(outputs, sentiment, weight=class_weights)

        # 2. TextGCN 权重 L2 正则化
        L2_reg = torch.tensor(0., requires_grad=True).to(device)
        for W in model.W_GCN:
            L2_reg = L2_reg + torch.sum(W ** 2)
        L_GCN = lambda_1 * L2_reg

        # 3. BiLSTM 权重 L1 稀疏化
        L1_reg = torch.tensor(0., requires_grad=True).to(device)
        for W in model.W_BiLSTM:
            L1_reg = L1_reg + torch.sum(torch.abs(W))
        L_BiLSTM = lambda_2 * L1_reg

        # 4. 动态图边权重变化惩罚项
        if model.prev_adj_matrix is not None:
            delta_A = model.adj_matrix - model.prev_adj_matrix
            L_dynamic = lambda_3 * torch.sum(delta_A ** 2)
        else:
            L_dynamic = torch.tensor(0.).to(device)

        # 5. 总损失函数
        L_total = L_bal + L_GCN + L_BiLSTM + L_dynamic

        L_total.backward()
        optimizer.step()

        total_loss += L_total.item()

    return total_loss / len(data_loader)

# 8. 评估代码
def evaluate(model, data_loader, device, build_graph, tokenizer, dataset_type="Validation"):
    model.eval()
    all_predicted = []
    all_sentiment = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)

            # 计算动态图
            corpus = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            edge_index, _ = build_graph(corpus, tokenizer)  # 获取 edge_index，忽略 word_index
            edge_index = edge_index.to(device)

            outputs = model(input_ids, attention_mask, edge_index)
            _, predicted = torch.max(outputs.data, 1)

            all_predicted.extend(predicted.cpu().numpy())
            all_sentiment.extend(sentiment.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_sentiment, all_predicted)
    precision = precision_score(all_sentiment, all_predicted)
    recall = recall_score(all_sentiment, all_predicted)
    f1 = f1_score(all_sentiment, all_predicted)

    # 打印评估指标
    print(f'{dataset_type} Accuracy: {accuracy:.4f}')
    print(f'{dataset_type} Precision: {precision:.4f}')
    print(f'{dataset_type} Recall: {recall:.4f}')
    print(f'{dataset_type} F1 Score: {f1:.4f}')

    # 绘制混淆矩阵
    cm = confusion_matrix(all_sentiment, all_predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{dataset_type} Confusion Matrix')
    plt.savefig(f'{dataset_type}_confusion_matrix.png')  # 保存图像
    plt.show()

    return accuracy, precision, recall, f1

# 9. 构建 TextGCN 的图结构
def build_graph(corpus, tokenizer):
    word_counts = Counter()
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        word_counts.update(tokens)

    word_index = {word: i for i, word in enumerate(word_counts.keys())}
    num_nodes = len(word_index)

    edges = []
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        for i in range(len(tokens) - 1):
            word1 = tokens[i]
            word2 = tokens[i + 1]
            index1 = word_index[word1]
            index2 = word_index[word2]
            edges.append((index1, index2))
            edges.append((index2, index1))  # 无向图

    # 去重
    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, word_index  # 返回 word_index

# 10. 文本清洗函数
def clean_text(text, slang_mapping):
    # 确保 text 是字符串类型
    text = str(text)

    # 使用 slang_mapping 替换网络用语
    for slang, standard in slang_mapping.items():
        text = text = text.replace(slang, standard)

    # 去除##之间的内容以及#
    text = re.sub(r'#.*?#', '', text)

    # 去除@提及部分
    text = re.sub(r'@\w+', '', text)

    # 去除超链接
    text = re.sub(r'http\S+', '', text)

    # 去除特殊字符，只保留中文
    text = "".join(re.findall(u'[\u4e00-\u9fa5]', str(text)))

    # 去除多余空格
    text = text.strip()

    return text

# 11. 主函数
def main():
    # 参数设置
    bert_model_name = '/mnt/bert-base-chinese'  # 修改为本地 BERT 模型路径
    gcn_hidden_dim = 256
    lstm_hidden_dim = 128
    num_classes = 2  # 情感类别：积极、消极
    max_len = 128
    batch_size = 32
    learning_rate = 1e-5
    epochs = 5
    embedding_dim = 768
    sentiment_conv_filters = 64  # 情感强度预测子网络卷积层滤波器数量
    sentiment_fc_dim = 32  # 情感强度预测子网络全连接层维度
    lambda_1 = 1e-5  # TextGCN 权重 L2 正则化系数
    lambda_2 = 1e-5  # BiLSTM 权重 L1 稀疏化系数
    lambda_3 = 1e-5  # 动态图边权重变化惩罚项系数
    step_size = 2  # 学习率衰减步长
    gamma = 0.1  # 学习率衰减因子
    validation_size = 0.1  # 验证集比例
    test_size = 0.1  # 测试集比例

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据准备
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    df = pd.read_csv('weibo_senti_100k.csv', encoding='utf-8')  # 修改为您的数据集路径
    df = df[['review', 'label']]  # 确保列名正确
    # df.columns = ['text', 'label']  # 移除这行

    # 划分训练集、验证集和测试集
    train_df, temp_df = train_test_split(df, test_size=(validation_size + test_size), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (validation_size + test_size), random_state=42)

    # 打印数据集大小
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 计算类别权重
    train_labels = train_df['label'].tolist()
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.tensor([total_samples / class_counts[c] for c in range(num_classes)], dtype=torch.float).to(device)

    # 构建图结构
    corpus = train_df['review'].tolist()  # 修改为 'review'
    edge_index, word_index = build_graph(corpus, tokenizer)  # 获取 edge_index 和 word_index
    edge_index = edge_index.to(device)
    vocab_size = len(word_index)

    # 定义 slang_mapping
    slang_mapping = {
        '233': '笑',  # 网络用语，表示大笑
        '666': '厉害', # 网络用语，表示很棒
        'orz': '失落', # 表情符号，表示沮丧
        'plz': '请',   # 英文缩写
        'thx': '谢谢',  # 英文缩写
        '泥垢': '你够了', # 方言谐音
        '表酱紫': '不要这样', # 卖萌说法
        '果咩': '抱歉', # 日语谐音
    }

    train_dataset = WeiboDataset(train_df, tokenizer, max_len, slang_mapping)
    val_dataset = WeiboDataset(val_df, tokenizer, max_len, slang_mapping)
    test_dataset = WeiboDataset(test_df, tokenizer, max_len, slang_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 模型初始化
    model = BERT_TextGCN_BiLSTM(bert_model_name, gcn_hidden_dim, lstm_hidden_dim, num_classes, vocab_size, embedding_dim,
                                 sentiment_conv_filters, sentiment_fc_dim, class_weights).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 记录损失和准确率
    train_losses = []
    val_accuracies = []

    # 训练
    best_val_accuracy = 0.0  # 初始化最佳验证集准确率
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device, build_graph, tokenizer, class_weights, lambda_1, lambda_2, lambda_3)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')

        # 评估验证集
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, build_graph, tokenizer, dataset_type="Validation")

        # 记录损失和准确率
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        # 如果验证集准确率提高，则保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型

        # 学习率衰减
        scheduler.step()

    # 绘制损失和准确率曲线图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.legend()

    plt.savefig('loss_accuracy_curves.png')  # 保存图像
    plt.show()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 在测试集上评估模型
    test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device, build_graph, tokenizer, dataset_type="Test")

if __name__ == '__main__':
    main()
