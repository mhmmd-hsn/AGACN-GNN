# Decoding Lingual EEG Signals using Graph Neural Networks

## 📌 Overview

This repository contains a reimplementation of the paper: [Decoding Bilingual EEG Signals With Complex Semantics Using Adaptive Graph Attention Convolutional Network](https://ieeexplore.ieee.org/document/10379024). The original paper introduces a novel approach to signal classification using GCNs. Our implementation replicates the methodology and experiments described in the paper, ensuring reproducibility and analysis of results.
## 🚀 Features

- Implements the core model architecture as described in the paper.
- Update dataset format in order to make it much easier to use.
- Offers visualization of graphs and training results.
- Modular and extensible codebase.

## 📂 Project Structure

```
├── AGACN.py                # Model class
├── connections.py          # connectivity calculator class
├── csv_creator.m           # data transformer
├── data_loader.py          # include preprocessing and trial extraction
├── main.py                 # main commands to start training
├── trainer.py              # train module
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── visualization.py        # plot siganls, connectivity, loss and accuracy plots
```

## 🛠 Installation

To install dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure you have the correct Python version (e.g., Python 3.8+).

## 📊 Dataset

We use the datasets from paper, because of problems with loading data.bdf files using python we have created csv_creator.m file to create csv format dataset; you can download the transformed dataset using [this link](https://drive.google.com/file/d/1Q7qYEF05y31NpP_LNnsWkDKFAZtAGdcf/view?usp=drive_link).
## 🔧 Usage

### Training the Model

To train the model, use the following command:

```bash
python main.py --train --epochs 100 --batch_size 32
```
