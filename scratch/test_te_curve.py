import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

df = pd.read_parquet('data/processed/cleaned.parquet')
te_train = df[(df['language'] == 'te') & (df['split'] == 'train')].copy()
te_test = df[(df['language'] == 'te') & (df['split'] == 'test')].copy()

te_train['y'] = (te_train['project_label'] != 'benign').astype(int)
te_test['y'] = (te_test['project_label'] != 'benign').astype(int)

y_test = te_test['y'].values

print(f"Total available Telugu train: {len(te_train)}, test: {len(te_test)}")
print(f"Train label balance: {te_train['y'].value_counts().to_dict()}")
print(f"Test label balance: {te_test['y'].value_counts().to_dict()}")

subsets = [25, 50, 100, 200, 300, len(te_train)]

print("\n--- Monolingual Telugu Data Scaling ---")
from sklearn.model_selection import train_test_split

for n in subsets:
    if n < len(te_train):
        sub, _ = train_test_split(
            te_train,
            train_size=n,
            stratify=te_train["y"],
            random_state=42,
        )
    else:
        sub = te_train.copy()
    
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(sub['text'])
    X_te = vec.transform(te_test['text'])
    
    k_folds = min(3, sub['y'].value_counts().min())
    if k_folds >= 2:
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', dual=False, random_state=42), cv=k_folds)
    else:
        clf = LinearSVC(C=1.0, class_weight='balanced', dual=False, random_state=42)
    clf.fit(X_tr, sub['y'].values)
    preds = clf.predict(X_te)
    
    f1 = f1_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    acc = accuracy_score(y_test, preds)
    n_pos = sub['y'].sum()
    print(f"N={n:3d} (actual {len(sub):3d}, threats={n_pos:2d}): Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
