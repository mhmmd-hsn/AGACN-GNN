import mne
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset



class ModelTrainer:
    def __init__(self, model, data, labels, num_folds=5, batch_size=80, learning_rate=0.0001, epochs=500):
        """Handles K-Fold cross-validation and training of the AGACN model."""
        self.model = model
        self.data = data
        self.labels = labels
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
    
    def train(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.data)):
            print(f"Training fold {fold+1}/{self.num_folds}")
            train_data, val_data = self.data[train_idx], self.data[val_idx]
            train_labels, val_labels = self.labels[train_idx], self.labels[val_idx]
            train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=self.batch_size, shuffle=False)
            for epoch in range(self.epochs):
                for inputs, labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")
