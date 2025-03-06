import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import numpy as np
from visualization import Visualization
from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, model, dataset, lr=0.0001, epochs=500, batch_size=80, num_folds=5, save_best=True, save_path="best_model.pth"):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        self.save_best = save_best
        self.save_path = save_path
        self.best_acc = 0  # Track best accuracy for model saving
        self.visualizer = Visualization()
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []  # Store probabilities for ROC curve
        self.best_fold_metrics = None  # Store the best fold's metrics
    
    def train(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(self.dataset)))):
            print(f"\nFold {fold + 1}/{self.num_folds}")
            
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
            
            fold_train_losses, fold_train_accs = [], []
            fold_val_losses, fold_val_accs = [], []
            best_fold_epoch_acc = 0
            
            for epoch in range(self.epochs):
                train_loss, correct, total = 0, 0, 0
                self.model.train()
                
                for feature_matrix, adjacency_matrix, labels in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(feature_matrix, adjacency_matrix)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                    
                train_acc = correct / total
                val_loss, val_acc, preds, labels, probs = self.validate(val_loader)
                
                fold_train_losses.append(train_loss / len(train_loader))
                fold_train_accs.append(train_acc)
                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                if self.save_best and val_acc > best_fold_epoch_acc:
                    best_fold_epoch_acc = val_acc
                    torch.save(self.model.state_dict(), f'best_model_fold{fold+1}.pth')
                
                if self.save_best and val_acc > self.best_acc:
                    self.best_acc = val_acc
                    torch.save(self.model.state_dict(), self.save_path)
                    self.best_fold_metrics = (fold_train_losses, fold_train_accs, fold_val_losses, fold_val_accs)
        
        if self.best_fold_metrics:
            self.visualizer.plot_loss_accuracy(*self.best_fold_metrics)
            

        self.plot_confusion_matrix()
        self.plot_roc_curve()

    def validate(self, val_loader):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for feature_matrix, adjacency_matrix, labels in val_loader:
                output = self.model(feature_matrix, adjacency_matrix)
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                
                probs = torch.exp(output)  # Convert log-softmax back to normal probabilities
                pred = probs.argmax(dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        self.all_preds.extend(all_preds)
        self.all_labels.extend(all_labels)
        self.all_probs.extend(all_probs)
        
        return val_loss / len(val_loader), correct / total, all_preds, all_labels, all_probs

    def plot_confusion_matrix(self):
        class_names = [str(i) for i in range(self.model.fc.out_features)]  # Assuming classification labels
        self.visualizer.plot_confusion_matrix(self.all_labels, self.all_preds, class_names)
        
    
    def plot_roc_curve(self):
        class_names = self.model.fc.out_features
        self.visualizer.plot_roc_curve(np.array(self.all_labels), np.array(self.all_probs), class_names)
        
