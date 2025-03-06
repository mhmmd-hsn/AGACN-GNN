import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import mne
import networkx as nx
from data_loader import EEGDataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

class Visualization:
    def __init__(self):
        sns.set(style="whitegrid")
    
    def plot_eeg_topomap(feature_matrix, adjacency_matrix, node_labels, threshold=0.1):
        """
        Plots EEG electrode positions and connectivity using MNE's built-in head model.
        """
        num_channels = len(node_labels)
        
        # Get standard EEG positions (10-20 system)
        montage = mne.channels.make_standard_montage("standard_1020")
        pos_dict = montage.get_positions()['ch_pos']
        
        # Extract 2D positions for available channels
        xy_positions = np.array([pos_dict[ch][:2] for ch in node_labels])

        # Normalize node sizes based on mean EEG activity
        node_sizes = feature_matrix.mean(axis=1)
        node_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min() + 1e-6) * 10  # Normalize
        

        # Plot EEG topomap (activity distribution)
        mne.viz.plot_topomap(node_sizes, xy_positions, cmap='coolwarm', contours=0, show=False)

        # Add channel labels manually
        for i, label in enumerate(node_labels):
            plt.text(xy_positions[i, 0], xy_positions[i, 1], label, fontsize=9, ha='center', va='center', color='black')

        # Create a graph for EEG connectivity
        G = nx.Graph()
        
        # Add nodes with positions
        for i, label in enumerate(node_labels):
            G.add_node(label, pos=xy_positions[i])

        # Add edges based on adjacency matrix
        for i in range(num_channels):
            for j in range(i + 1, num_channels):  # Avoid duplicate edges
                weight = adjacency_matrix[i, j]
                if weight > threshold:  # Only add strong connections
                    G.add_edge(node_labels[i], node_labels[j], weight=weight)

        # Overlay connectivity graph
        pos = nx.get_node_attributes(G, 'pos')
        edges = G.edges(data=True)
        edge_widths = [d['weight'] * 2 for (u, v, d) in edges]  # Scale edge thickness
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='red', alpha=0.8)

        plt.show()

    
    def plot_loss_accuracy(self, train_losses, val_losses, train_accs, val_accs):
        """ Plots training & validation loss and accuracy """
        epochs = range(1, len(train_losses) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax2.plot(epochs, train_accs, 'b--', label='Train Acc')
        ax2.plot(epochs, val_accs, 'r--', label='Val Acc')
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        ax1.legend(loc='upper right')
        ax2.legend(loc='lower right')
        plt.title("Training and Validation Loss & Accuracy")
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, num_classes):
        """ Plots ROC curve for multi-class classification """
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Multi-class Classification")
        plt.legend()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """ Plots confusion matrix """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
    
    def print_classification_report(self, y_true, y_pred, class_names):
        """ Prints a detailed classification report """
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("Classification Report:\n", report)
