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
        self.vocab_size = tokenizer.vocab_size  # 使用tokenizer的词汇表大小

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

        # 验证和修正输入ID
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            invalid_mask = (input_ids < 0) | (input_ids >= self.vocab_size)
            invalid_count = invalid_mask.sum().item()
            print(f"样本 {index}: 发现 {invalid_count} 个无效输入ID，词汇表大小: {self.vocab_size}")
            
            # 记录无效ID的位置和值
            if invalid_count < 10:  # 只打印少量无效ID，避免过多输出
                invalid_indices = torch.nonzero(invalid_mask, as_tuple=True)
                for i, j in zip(*invalid_indices):
                    print(f"无效ID位置: [{i}, {j}], 值: {input_ids[i, j]}")
                    # 尝试解码无效ID，帮助诊断问题
                    if input_ids[i, j] < self.vocab_size:
                        print(f"无效ID对应的token: {self.tokenizer.convert_ids_to_tokens([input_ids[i, j].item()])}")
            
            # 修正超出范围的ID为[UNK]标记 (通常是100)
            input_ids[input_ids < 0] = 100
            input_ids[input_ids >= self.vocab_size] = 100

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
        # 验证输入ID是否在词汇表范围内
        if torch.any(input_ids >= self.vocab_size) or torch.any(input_ids < 0):
            invalid_mask = (input_ids >= self.vocab_size) | (input_ids < 0)
            invalid_count = invalid_mask.sum().item()
            print(f"警告: 发现 {invalid_count} 个无效输入ID，词汇表大小: {self.vocab_size}")
            
            # 记录无效ID的位置和值
            if invalid_count < 10:  # 只打印少量无效ID，避免过多输出
                invalid_indices = torch.nonzero(invalid_mask, as_tuple=True)
                for i, j in zip(*invalid_indices):
                    print(f"无效ID位置: [{i}, {j}], 值: {input_ids[i, j]}")
                    # 尝试解码无效ID，帮助诊断问题
                    if input_ids[i, j] < self.vocab_size:
                        print(f"无效ID对应的token: {self.bert.tokenizer.convert_ids_to_tokens([input_ids[i, j].item()])}")
            
            # 修正超出范围的ID为[UNK]标记 (通常是100)
            input_ids[input_ids < 0] = 100
            input_ids[input_ids >= self.vocab_size] = 100

        # BERT Embedding
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.last_hidden_state

        # TextGCN
        x = self.embedding(input_ids)  # input_ids 作为节点特征
        # 残差连接
        x_initial = self.input_linear(x)  # 将输入特征映射到 GCN 隐藏层维度
        
        # 确保图结构有效
        edge_index = ensure_valid_graph(edge_index, x.size(0))
        
        x = F.relu(self.gcn1(x, edge_index))
        x = x + x_initial  # 残差连接
        
        # 再次确保图结构有效
        edge_index = ensure_valid_graph(edge_index, x.size(0))
        
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

# 确保图结构有效（无孤立节点）
def ensure_valid_graph(edge_index, num_nodes):
    # 检查图是否包含孤立节点
    if edge_index.numel() > 0:
        # 计算每个节点的度
        deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        src, dst = edge_index
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.long, device=edge_index.device))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.long, device=edge_index.device))
        
        # 检查是否有度为0的节点
        if torch.any(deg == 0):
            print(f"发现 {torch.sum(deg == 0)} 个孤立节点，添加自环")
            # 为每个孤立节点添加自环
            isolated_nodes = torch.nonzero(deg == 0, as_tuple=True)[0]
            if len(isolated_nodes) > 0:
                self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
                edge_index = torch.cat([edge_index, self_loops], dim=1)
    else:
        # 如果没有边，则创建一个自环图
        edge_index = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device).repeat(2, 1)
        print(f"创建了自环图，节点数: {num_nodes}")
    
    return edge_index

# 7. 训练代码
def train(model, data_loader, optimizer, device, build_graph, tokenizer, class_weights, lambda_1, lambda_2, lambda_3):
    model.train()
    total_loss = 0
    batch_count = 0
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        sentiment = batch['sentiment']

        # 验证输入ID
        if torch.any(input_ids < 0) or torch.any(input_ids >= tokenizer.vocab_size):
            invalid_mask = (input_ids < 0) | (input_ids >= tokenizer.vocab_size)
            invalid_count = invalid_mask.sum().item()
            print(f"Batch {batch_count}: 发现 {invalid_count} 个无效输入ID，词汇表大小: {tokenizer.vocab_size}")
            
            # 记录无效ID的位置和值
            if invalid_count < 10:  # 只打印少量无效ID，避免过多输出
                invalid_indices = torch.nonzero(invalid_mask, as_tuple=True)
                for i, j in zip(*invalid_indices):
                    print(f"无效ID位置: [{i}, {j}], 值: {input_ids[i, j]}")
                    # 尝试解码无效ID，帮助诊断问题
                    if input_ids[i, j] < tokenizer.vocab_size:
                        print(f"无效ID对应的token: {tokenizer.convert_ids_to_tokens([input_ids[i, j].item()])}")
            
            # 修正无效ID为[UNK]标记
            input_ids[input_ids < 0] = tokenizer.unk_token_id
            input_ids[input_ids >= tokenizer.vocab_size] = tokenizer.unk_token_id

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        sentiment = sentiment.to(device)

        # 计算动态图
        corpus = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        edge_index, word_index = build_graph(corpus, tokenizer)  # 获取 edge_index，忽略 word_index
        
        # 验证 edge_index
        if edge_index.numel() > 0:
            if torch.max(edge_index) >= len(word_index):
                print(f"Batch {batch_count}: 发现 edge_index 中的值 >= 词汇表大小 ({len(word_index)})")
                print(f"edge_index 最大值: {torch.max(edge_index)}")
                edge_index = torch.clamp(edge_index, 0, len(word_index) - 1)
            if torch.min(edge_index) < 0:
                print(f"Batch {batch_count}: 发现 edge_index 中的负值")
                print(f"edge_index 最小值: {torch.min(edge_index)}")
                edge_index = torch.clamp(edge_index, 0, len(word_index) - 1)

        edge_index = ensure_valid_graph(edge_index, len(word_index))
        edge_index = edge_index.to(device)

        optimizer.zero_grad()
        try:
            outputs = model(input_ids, attention_mask, edge_index)
        except Exception as e:
            print(f"Batch {batch_count}: 前向传播错误: {e}")
            print(f"输入ID范围: min={torch.min(input_ids)}, max={torch.max(input_ids)}")
            print(f"edge_index范围: min={torch.min(edge_index)}, max={torch.max(edge_index)}")
            print(f"词汇表大小: {model.vocab_size}")
            
            # 尝试识别问题来源
            try:
                # 检查BERT部分
                _ = model.bert(input_ids, attention_mask=attention_mask)
                print("BERT部分运行正常")
                
                # 检查GCN部分
                x = model.embedding(input_ids)
                x_initial = model.input_linear(x)
                
                # 手动执行GCN归一化，确保处理了孤立节点
                edge_index_gcn = ensure_valid_graph(edge_index, x.size(0))
                from torch_geometric.utils import add_self_loops, degree
                edge_index_gcn, _ = add_self_loops(edge_index_gcn, num_nodes=x.size(0))
                row, col = edge_index_gcn
                deg = degree(col, x.size(0), dtype=x.dtype)
                deg_inv_sqrt = deg.pow_(-0.5)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)  # 处理除零情况
                edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                x_gcn = F.relu(model.gcn1(x, edge_index_gcn, edge_weight))
                print("GCN第一层运行正常")
                
                # 对第二层执行相同的检查
                edge_index_gcn = ensure_valid_graph(edge_index, x_gcn.size(0))
                edge_index_gcn, _ = add_self_loops(edge_index_gcn, num_nodes=x_gcn.size(0))
                row, col = edge_index_gcn
                deg = degree(col, x_gcn.size(0), dtype=x_gcn.dtype)
                deg_inv_sqrt = deg.pow_(-0.5)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)  # 处理除零情况
                edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                x_gcn = F.relu(model.gcn2(x_gcn, edge_index_gcn, edge_weight))
                print("GCN第二层运行正常")
                
                # 检查情感强度预测
                sentiment_input = x_gcn.permute(0, 2, 1)
                sentiment_output = F.relu(model.sentiment_conv1(sentiment_input))
                sentiment_output = F.relu(model.sentiment_conv2(sentiment_output))
                sentiment_output = sentiment_output.view(sentiment_output.size(0), -1)
                sentiment_output = F.relu(model.sentiment_fc(sentiment_output))
                sentiment_intensity = model.sentiment_output(sentiment_output)
                sentiment_intensity = torch.sigmoid(sentiment_intensity)
                print("情感强度预测运行正常")
                
                # 检查BiLSTM
                bilstm_output = model.bilstm(x_gcn, sentiment_intensity)
                print("BiLSTM运行正常")
                
                # 检查注意力机制
                attended_output = model.attention(bilstm_output, sentiment_intensity)
                print("注意力机制运行正常")
                
                # 检查DNN
                _ = model.dnn(attended_output, model.class_weights)
                print("DNN运行正常")
                
                print("问题可能出在这些组件之间的交互")
            except Exception as inner_e:
                print(f"组件测试失败: {inner_e}")
            
            continue  # 跳过当前批次

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
        batch_count += 1

    return total_loss / max(1, batch_count)

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

            # 验证输入ID
            if torch.any(input_ids < 0) or torch.any(input_ids >= tokenizer.vocab_size):
                print(f"评估阶段: 发现无效输入ID")
                input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)

            # 计算动态图
            corpus = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            edge_index, word_index = build_graph(corpus, tokenizer)  # 获取 edge_index，忽略 word_index
            
            # 验证 edge_index
            if edge_index.numel() > 0:
                if torch.max(edge_index) >= len(word_index):
                    print(f"评估阶段: 发现 edge_index 中的值 >= 词汇表大小 ({len(word_index)})")
                    edge_index = torch.clamp(edge_index, 0, len(word_index) - 1)
                if torch.min(edge_index) < 0:
                    print(f"评估阶段: 发现 edge_index 中的负值")
                    edge_index = torch.clamp(edge_index, 0, len(word_index) - 1)

            edge_index = ensure_valid_graph(edge_index, len(word_index))
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

    # 确保词汇表包含所有特殊token
    special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
    for token in special_tokens:
        if token not in word_counts:
            word_counts[token] = 1  # 添加特殊token

    word_index = {word: i for i, word in enumerate(word_counts.keys())}
    num_nodes = len(word_index)

    edges = []
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        for i in range(len(tokens) - 1):
            word1 = tokens[i]
            word2 = tokens[i + 1]
            if word1 in word_index and word2 in word_index:
                index1 = word_index[word1]
                index2 = word_index[word2]
                edges.append((index1, index2))
                edges.append((index2, index1))  # 无向图

    # 去重
    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    # 处理孤立节点
    edge_index = ensure_valid_graph(edge_index, num_nodes)

    return edge_index, word_index

# 10. 文本清洗函数
def clean_text(text, slang_mapping):
    # 确保 text 是字符串类型
    text = str(text)

    # 使用 slang_mapping 替换网络用语
    for slang, standard in slang_mapping.items():
        text = text.replace(slang, standard)

    # 去除##之间的内容以及#
    text = re.sub(r'#.*?#', '', text)

    # 去除@提及部分
    text = re.sub(r'@\w+', '', text)

    # 去除超链接
    text = re.sub(r'http\S+', '', text)

    # 保留中文、数字和基本标点符号
    text = "".join(re.findall(u'[\u4e00-\u9fa50-9，。！？、：；,.?!:;]', str(text)))

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
    print(f"类别权重: {class_weights}")

    # 网络用语映射表
    slang_mapping = {    # 网络用语映射表
    slang_mapping = {
        'yyds': '永远的神',
        '绝绝子': '太绝了',
        'awsl': '啊我死了',
        '集美': '姐妹',
        '杠精': '爱抬杠的人',
        '佛系': '无所谓',
        '躺平': '放弃努力',
        '内卷': '过度竞争',
        '摆烂': '破罐子破摔',
        'emo': '情绪低落',
        '奥利给': '加油',
        '好家伙': '表示惊讶',
        '666': '很厉害',
        '瑟瑟发抖': '害怕',
        '吃瓜': '看热闹',
        '凡尔赛': '炫耀',
        'yygq': '阴阳怪气',
        '打工人': '劳动者',
        '干饭人': '热爱吃饭的人',
        '绝了': '太棒了',
        '笑死我了': '太搞笑了',
        '裂开': '心态崩了',
        '淦': '干',
        '躺赢': '不努力却成功',
        'CP': '情侣组合',
        '嗑CP': '支持情侣',
        '雷人': '令人震惊',
        '种草': '推荐',
        '拔草': '不推荐',
        '上头': '沉迷',
        '冲': '行动',
        '奥利给': '加油',
        '好家伙': '表示惊讶',
        '绝绝子': '太绝了',
        'YYDS': '永远的神',
        'NB': '厉害',
        'SB': '傻子',
        '佛系': '无所谓',
        '躺平': '放弃努力',
        '内卷': '过度竞争',
        '摆烂': '破罐子破摔',
        'emo': '情绪低落',
        'yyds': '永远的神',
        'awsl': '啊我死了',
        '集美': '姐妹',
        '杠精': '爱抬杠的人',
        '666': '很厉害',
        '瑟瑟发抖': '害怕',
        '吃瓜': '看热闹',
        '凡尔赛': '炫耀',
        'yygq': '阴阳怪气',
        '打工人': '劳动者',
        '干饭人': '热爱吃饭的人',
        '绝了': '太棒了',
        '笑死我了': '太搞笑了',
        '裂开': '心态崩了',
        '淦': '干',
        '躺赢': '不努力却成功',
        'CP': '情侣组合',
        '嗑CP': '支持情侣',
        '雷人': '令人震惊',
        '种草': '推荐',
        '拔草': '不推荐',
        '上头': '沉迷',
        '冲': '行动',
        '奥利给': '加油',
        '好家伙': '表示惊讶',
        '绝绝子': '太绝了',
        'YYDS': '永远的神',
        'NB': '厉害',
        'SB': '傻子',
        '佛系': '无所谓',
        '躺平': '放弃努力',
        '内卷': '过度竞争',
        '摆烂': '破罐子破摔',
        'emo': '情绪低落',
        'yyds': '永远的神',
        'awsl': '啊我死了',
        '集美': '姐妹',
        '杠精': '爱抬杠的人',
        '666': '很厉害',
        '瑟瑟发抖': '害怕',
        '吃瓜': '看热闹',
        '凡尔赛': '炫耀',
        'yygq': '阴阳怪气',
        '打工人': '劳动者',
        '干饭人': '热爱吃饭的人',
        '绝了': '太棒了',
        '笑死我了': '太搞笑了',
        '裂开': '心态崩了',
        '淦': '干',
        '躺赢': '不努力却成功',
        'CP': '情侣组合',
        '嗑CP': '支持情侣',
        '雷人': '令人震惊',
        '种草': '推荐',
        '拔草': '不推荐',
        '上头': '沉迷',
        '冲': '行动',
        '奥利给': '加油',
        '好家伙': '表示惊讶',
        '绝绝子': '太绝了',
        'YYDS': '永远的神',
        'NB': '厉害',
        'SB': '傻子',
    }

    # 创建数据集和数据加载器
    train_dataset = WeiboDataset(train_df, tokenizer, max_len, slang_mapping)
    val_dataset = WeiboDataset(val_df, tokenizer, max_len, slang_mapping)
    test_dataset = WeiboDataset(test_df, tokenizer, max_len, slang_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型
    model = BERT_TextGCN_BiLSTM(
        bert_model_name=bert_model_name,
        gcn_hidden_dim=gcn_hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        num_classes=num_classes,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        sentiment_conv_filters=sentiment_conv_filters,
        sentiment_fc_dim=sentiment_fc_dim,
        class_weights=class_weights
    ).to(device)

    # 冻结BERT的部分参数
    for param in list(model.bert.parameters())[:-10]:  # 只微调最后几层
        param.requires_grad = False

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 训练循环
    best_val_f1 = 0.0
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 30)

        # 训练阶段
        train_loss = train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            build_graph=build_graph,
            tokenizer=tokenizer,
            class_weights=class_weights,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3
        )
        print(f'Training Loss: {train_loss:.4f}')

        # 验证阶段
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            build_graph=build_graph,
            tokenizer=tokenizer,
            dataset_type="Validation"
        )

        # 学习率衰减
        scheduler.step()

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with F1 score: {best_val_f1:.4f}')

        print()

    # 在测试集上评估最佳模型
    print('Evaluating on test set...')
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy, test_precision, test_recall, test_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        build_graph=build_graph,
        tokenizer=tokenizer,
        dataset_type="Test"
    )

    print('Training and evaluation completed!')

if __name__ == "__main__":
    main()
