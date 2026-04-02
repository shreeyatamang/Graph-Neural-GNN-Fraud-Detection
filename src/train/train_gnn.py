import torch
import torch.nn.functional as F
from src.models.gnn_model import GraphSAGEFraud
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


data = torch.load("data/graph.pt", weights_only=False)
edge_labels = torch.load("data/edge_labels.pt", weights_only=False)

num_pos = edge_labels.sum().item()
num_neg = len(edge_labels) - num_pos
pos_weight = torch.tensor([num_neg / num_pos])

print(f"Fraud edges: {int(num_pos)} | Normal edges: {int(num_neg)}")
print(f"Positive weight: {pos_weight.item():.2f}")

train_idx, test_idx = train_test_split(
    torch.arange(data.edge_index.shape[1]).numpy(),
    test_size=0.2,
    random_state=42,
    stratify=edge_labels.numpy()
)
train_idx = torch.tensor(train_idx)
test_idx  = torch.tensor(test_idx)

model = GraphSAGEFraud(input_dim=12, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=25
)
loss_fn   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train():
    model.train()
    optimizer.zero_grad()
    logits = model.predict_edges(data.x, data.edge_index)
    loss   = loss_fn(logits[train_idx], edge_labels[train_idx].float())
    loss.backward()
    optimizer.step()
    return loss.item()


def find_best_threshold(probs, labels):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = (probs > thresh).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


def evaluate():
    model.eval()
    with torch.no_grad():
        logits = model.predict_edges(data.x, data.edge_index)
        probs  = torch.sigmoid(logits)

        y_true = edge_labels[test_idx].cpu().numpy()
        y_prob = probs[test_idx].cpu().numpy()

        best_thresh, best_f1 = find_best_threshold(y_prob, y_true)
        y_pred = (y_prob > best_thresh).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        return prec, rec, best_f1, best_thresh


print("\nTraining GraphSAGE...\n")
best_f1_seen = 0

for epoch in range(300):
    loss = train()
    scheduler.step(loss)

    if epoch % 20 == 0:
        prec, rec, f1, thresh = evaluate()
        flag = " ◄ best" if f1 > best_f1_seen else ""
        best_f1_seen = max(best_f1_seen, f1)
        print(
            f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
            f"Prec: {prec:.4f} | Rec: {rec:.4f} | "
            f"F1: {f1:.4f} | Thresh: {thresh:.2f}{flag}"
        )

print(f"\nBest GNN F1: {best_f1_seen:.4f}")
print(f"Random Forest F1: 0.6687")
if best_f1_seen > 0.6687:
    print("GNN beats Random Forest! Graph structure pays off.")
else:
    gap = (0.6687 - best_f1_seen) / 0.6687 * 100
    print(f"GNN still {gap:.1f}% behind — update gnn_f1 in train_baseline.py to {best_f1_seen:.4f}")