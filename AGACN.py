import torch
import torch.nn as nn
import torch.nn.functional as F

class AGACNCell(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', dropout=0.05, use_bias=True):
        super(AGACNCell, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        self.dropout = nn.Dropout(dropout)  
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("Invalid activation function. Use 'relu' or 'tanh'.")

    def forward(self, feature_matrix, adjacency_matrix):
        
        out = torch.matmul(adjacency_matrix, feature_matrix)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:  
            out += self.bias

        out = self.dropout(out)
        return self.activation(out)  

class AGACN(nn.Module):
    def __init__(self, num_timepoints=2500, num_classes=4):
        super(AGACN, self).__init__()

        # AGACN Layers with Correct Activations
        self.agacn1 = AGACNCell(num_timepoints, 126, activation='relu')
        self.agacn2 = AGACNCell(126, 64, activation='tanh')  
        self.agacn3 = AGACNCell(64, 132, activation='relu')

        self.fc = nn.Linear(64 * 132, num_classes)


    def count_parameters(self):
        """
        Prints the total number of trainable parameters in the model
        and the number of parameters per layer.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🔥 Total Trainable Parameters: {total_params:,}\n")

        print("🔍 Layer-wise Parameter Breakdown:")
        print("=" * 40)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name:<30} {param.numel():,} parameters")
        print("=" * 40)


    def forward(self, feature_matrix, adjacency_matrix):
        # Pass through AGACN layers
        out1 = self.agacn1(feature_matrix, adjacency_matrix)
        out2 = self.agacn2(out1, adjacency_matrix)
        out3 = self.agacn3(out2, adjacency_matrix)


        # Cross-Fusion Attention (CFA) Layer
        cfa_out = torch.matmul(out2.transpose(1, 2), out3)  # (H(2))^T * H(3)
        
        cfa_out = cfa_out.view(cfa_out.shape[0], -1)  # Flatten

        # Fully Connected Layer
        out = self.fc(cfa_out)
        return out