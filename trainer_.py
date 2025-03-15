import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from visualization import Visualization
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np




class Trainer:
    def __init__(self, model, dataset, lr=0.0001, epochs=500, batch_size=80, save_best=True, save_path="best_model.pth"):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        self.save_best = save_best
        self.save_path = save_path
        self.best_acc = 0  # Track best accuracy for model saving
        self.visualizer = Visualization()
        self.all_preds = []
        self.all_labels = []
        self.best_results = pd.DataFrame(columns=["Train Accuracy", "Valid Accuracy", "Test Accuracy", "Epoch Number"])

    def train(self):
        train_idx, val_idx, test_idx = self.custom_split(self.dataset)
        
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        test_subset = Subset(self.dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        best_epoch_acc = 0
        for epoch in range(self.epochs):
            train_loss = 0
            all_train_preds, all_train_labels = [], []
            self.model.train()
            
            for feature_matrix, adjacency_matrix, labels in train_loader:
                self.optimizer.zero_grad()
                output = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                pred = torch.argmax(output, dim=-1)
                all_train_preds.extend(pred.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            val_loss, val_acc = self.validate(val_loader)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if self.save_best and val_acc > best_epoch_acc:
                best_epoch_acc = val_acc
                self.best_results = self.add_row([train_acc, val_acc, 0.0, epoch])
                
        test_loss, test_acc = self.validate(test_loader)
        print(f'Test Accuracy: {test_acc:.4f}')
        self.best_results.iloc[-1, 2] = test_acc
        
    def validate(self, val_loader):
        self.model.eval()
        device = next(self.model.parameters()).device
        self.model.to(device)

        val_loss, all_preds, all_labels = 0, [], []

        with torch.no_grad():
            for features, adj, labels in val_loader:
                features, adj, labels = features.to(device), adj.to(device), labels.to(device)

                output = self.model(features, adj)
                val_loss += self.criterion(output, labels).item()
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return val_loss / len(val_loader), accuracy_score(all_labels, all_preds)

    def add_row(self, data):
        new_row = pd.DataFrame([data], columns=self.best_results.columns)
        return pd.concat([self.best_results, new_row], ignore_index=True)

    def custom_split(self,dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        labels = np.array(dataset.labels)
        unique_classes = np.unique(labels)
        train_idx, val_idx, test_idx = [], [], []
        
        np.random.seed(random_state)
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            np.random.shuffle(cls_indices)
            
            train_size = int(len(cls_indices) * train_ratio)
            val_size = int(len(cls_indices) * val_ratio)
            
            train_idx.extend(cls_indices[:train_size])
            val_idx.extend(cls_indices[train_size:train_size + val_size])
            test_idx.extend(cls_indices[train_size + val_size:])
        
        return train_idx, val_idx, test_idx
