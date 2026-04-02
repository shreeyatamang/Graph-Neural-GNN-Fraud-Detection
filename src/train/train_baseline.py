import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


data = torch.load("data/graph.pt", weights_only=False)
edge_labels = torch.load("data/edge_labels.pt", weights_only=False)


src_nodes = data.edge_index[0]
dst_nodes = data.edge_index[1]

src_feats = data.x[src_nodes].numpy()  
dst_feats = data.x[dst_nodes].numpy()   


X = np.concatenate([src_feats, dst_feats], axis=1)  
y = edge_labels.numpy()

print(f"Feature matrix shape: {X.shape}")
print(f"Fraud rate: {y.mean():.2%}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Fraud in test: {y_test.sum()} / {len(y_test)}\n")



def evaluate_model(name, y_true, y_pred):
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    print(f"{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print()
    return f1


results = {}


print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Logistic Regression'] = evaluate_model("Logistic Regression", y_test, y_pred_lr)


print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = evaluate_model("Random Forest", y_test, y_pred_rf)



print("Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,   
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results['XGBoost'] = evaluate_model("XGBoost", y_test, y_pred_xgb)



gnn_f1 = 0.4748 

print("=" * 50)
print("  FINAL COMPARISON")
print("=" * 50)
print(f"  {'Model':<25} {'F1':>8}")
print(f"  {'─'*33}")
for model_name, f1 in results.items():
    print(f"  {model_name:<25} {f1:>8.4f}")
print(f"  {'GNN (graph-aware)':<25} {gnn_f1:>8.4f}  ◄ your model")
print("=" * 50)

best_baseline = max(results.values())
if gnn_f1 > best_baseline:
    lift = ((gnn_f1 - best_baseline) / best_baseline) * 100
    print(f"\n  GNN beats best baseline by {lift:.1f}%")
    print("  Graph structure is adding real value!")
else:
    gap = ((best_baseline - gnn_f1) / best_baseline) * 100
    print(f"\n  Best baseline beats GNN by {gap:.1f}%")
    print("  The graph structure needs more work to pay off.")