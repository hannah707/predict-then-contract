import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mlpRegressor1(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor1, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.initialize_weights(seed)

    def forward(self, x):
        out = self.fc(x)
        return out

    def initialize_weights(self,seed):
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

                    
class mlpRegressor3(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        # self.bn1 = nn.BatchNorm1d(hidden_dim1) 
        # self.bn2 = nn.BatchNorm1d(hidden_dim2) 
        self.initialize_weights(seed)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = self.bn1(out)
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        # out = # self.bn2(out)
        out = self.dropout2(out)
        out = F.relu(self.fc3(out))
        return out

    def initialize_weights(self,seed):
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
              


class mlpRegressor6(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor6, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc6 = nn.Linear(hidden_dim2, output_dim) #compare with the attention
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.initialize_weights(seed)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.dropout1(out)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.dropout2(out)
        out = F.relu(self.fc5(out))
        out = self.dropout3(out)
        out = F.relu(self.fc6(out))
        return out

    def initialize_weights(self,seed):
        # Custom weight initialization with fixed seed
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class mlpRegressor12(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor12, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc4 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc6 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc7 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc8 = nn.Linear(hidden_dim2, hidden_dim2*2)
        self.fc9 = nn.Linear(hidden_dim2*2, hidden_dim2*2)
        self.fc10 = nn.Linear(hidden_dim2*2, hidden_dim2*2)
        self.fc11 = nn.Linear(hidden_dim2*2, hidden_dim2*2)
        self.fc12 = nn.Linear(hidden_dim2*2, output_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.dropout5 = nn.Dropout(0.1)
        self.initialize_weights(seed)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.dropout1(out)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.dropout2(out)
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        out = self.dropout3(out)
        out = F.relu(self.fc7(out))
        out = F.relu(self.fc8(out))
        out = self.dropout4(out)
        out = F.relu(self.fc9(out))
        out = F.relu(self.fc10(out))
        out = self.dropout5(out)
        out = F.relu(self.fc11(out))
        out = F.relu(self.fc12(out))
        return out

    def initialize_weights(self,seed):
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)



class rnnRegressor6(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, num_rnn_layers=1, seed=2024):
        super(rnnRegressor6, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim1, num_rnn_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, output_dim)
        # self.fc6 = nn.Linear(hidden_dim2, output_dim) 
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_dim1

        self.initialize_weights(seed)

    def forward(self, x):
        h0 = torch.zeros(self.num_rnn_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = F.relu(out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.dropout2(out)
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        return out
    
    def initialize_weights(self,seed):
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) 
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class AttentionModule(nn.Module):
    def __init__(self, input_size, attention_size):
        super(AttentionModule, self).__init__()
        self.W_q = nn.Linear(input_size, attention_size)
        self.W_k = nn.Linear(input_size, attention_size)
        self.W_v = nn.Linear(input_size, attention_size)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = torch.sqrt(torch.FloatTensor([attention_size])).to(device)

    def forward(self, input):
        Q = self.W_q(input)
        K = self.W_k(input)
        V = self.W_v(input)

        attention_scores = torch.matmul(Q, K.t()) / self.scale
        attention_weights = self.softmax(attention_scores)

        output = torch.matmul(attention_weights, V)
    
        return output
    
class mlpRegressor1atti(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor1atti, self).__init__()
        self.attention = AttentionModule(input_dim,input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.initialize_weights(seed)

    def forward(self, x):
        out = self.attention(x)
        out = self.fc(out)
        return out

    def initialize_weights(self,seed):
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

                    

class mlpRegressor6atti(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor6atti, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc6 = nn.Linear(hidden_dim2, output_dim) #compare with the attention
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        # self.bn1 = nn.BatchNorm1d(hidden_dim1) 
        # self.bn2 = nn.BatchNorm1d(hidden_dim2) 
        self.attention = AttentionModule(input_dim, input_dim)
        self.initialize_weights(seed)

    def forward(self, x):
        out = self.attention(x)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.dropout1(out)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.dropout2(out)
        out = F.relu(self.fc5(out))
        out = self.dropout3(out)
        out = F.relu(self.fc6(out))
        return out

    def initialize_weights(self,seed):
        # Custom weight initialization with fixed seed
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

					

class mlpRegressor6attires(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor6attires, self).__init__()
        self.fc1 = nn.Linear(input_dim+input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc6 = nn.Linear(hidden_dim2, output_dim) #compare with the attention
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.attention = AttentionModule(input_dim, input_dim)
        self.initialize_weights(seed)

    def forward(self, x):
        out = self.attention(x)
        out = torch.cat((x,out),dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.dropout1(out)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.dropout2(out)
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        return out

    def initialize_weights(self,seed):
        # Custom weight initialization with fixed seed
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class mlpRegressor6attires2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, seed=2024):
        super(mlpRegressor6attires2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1+input_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc6 = nn.Linear(hidden_dim2, output_dim) #compare with the attention
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.attention = AttentionModule(input_dim, input_dim)
        self.initialize_weights(seed)

    def forward(self, x):
        att_out = self.attention(x)
        mlp_out = F.relu(self.fc1(x))
        out = torch.cat((att_out,mlp_out),dim=1)
        out = F.relu(self.fc2(out))
        out = self.dropout1(out)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.dropout2(out)
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        return out

    def initialize_weights(self,seed):
        # Custom weight initialization with fixed seed
        torch.manual_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)