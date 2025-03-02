import torch
import torch.nn as nn
import torch.nn.functional as F

class AGACNCell(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', dropout=0.05):
        super(AGACNCell, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.dropout = nn.Dropout(dropout)  # Apply dropout
        
        # Set activation function dynamically
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Invalid activation function. Use 'relu' or 'tanh'.")

    def forward(self, feature_matrix, adjacency_matrix):
        # Graph Convolution
        out = torch.matmul(adjacency_matrix, feature_matrix)
        out = torch.matmul(out, self.weight)
        out = self.dropout(out)  # Apply dropout
        return self.activation(out)  # Apply correct activation function

class AGACN(nn.Module):
    def __init__(self, num_channels=12, num_timepoints=2500, num_classes=4):
        super(AGACN, self).__init__()

        # AGACN Layers with Correct Activations
        self.agacn1 = AGACNCell(num_timepoints, 126, activation='relu')
        self.agacn2 = AGACNCell(126, 64, activation='tanh')  
        self.agacn3 = AGACNCell(64, 132, activation='relu')

        # Fully Connected Layer + Softmax
        self.fc = nn.Linear(132, num_classes)

    def forward(self, feature_matrix, adjacency_matrix):
        # Pass through AGACN layers
        out1 = self.agacn1(feature_matrix, adjacency_matrix)
        out2 = self.agacn2(out1, adjacency_matrix)
        out3 = self.agacn3(out2, adjacency_matrix)

        # Cross-Fusion Attention (CFA) Layer
        cfa_out = torch.matmul(out2.transpose(1, 2), out3)  # (H(2))^T * H(3)
        # Fully Connected Layer
        out = self.fc(cfa_out)
        return F.log_softmax(out, dim=1)



class Trainer:
    def __init__(self, model, lr=0.0001, epochs=500, batch_size=80):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    
    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for feature_matrix, adjacency_matrix, labels in train_loader:
                self.optimizer.zero_grad()
                output = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            val_loss, val_acc = self.validate(val_loader)
            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for feature_matrix, adjacency_matrix, labels in val_loader:
                output = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return val_loss / len(val_loader), correct / total