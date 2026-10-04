import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy import sparse
from src.features.intent.intent_extractor import extract_intent_features

df = pd.read_parquet('data/processed/cleaned.parquet')
ind = df[df['source_dataset'] == 'indian_scam'].copy()
ind['category'] = ind['scam_category'].fillna('legitimate')

def map_taxonomy(cat):
    if cat == 'police_digital_arrest': return 'digital_arrest'
    elif cat == 'police_blackmail': return 'extortion_blackmail'
    elif cat in ('bank_kyc', 'aadhaar'): return 'kyc_banking_identity'
    elif cat == 'lottery': return 'lottery_reward'
    elif cat in ('amazon', 'relative'): return 'impersonation_scam'
    elif cat == 'legitimate': return 'legitimate'
    return 'other'

ind['taxonomy'] = ind['category'].apply(map_taxonomy)

train_df = ind[ind['split'] == 'train']
val_df = ind[ind['split'] == 'validation']
test_df = ind[ind['split'] == 'test']

vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
X_train_text = vec.fit_transform(train_df['text'])
X_val_text = vec.transform(val_df['text'])
X_test_text = vec.transform(test_df['text'])

X_train_intent = sparse.csr_matrix(np.array([extract_intent_features(t) for t in train_df['text']]))
X_val_intent = sparse.csr_matrix(np.array([extract_intent_features(t) for t in val_df['text']]))
X_test_intent = sparse.csr_matrix(np.array([extract_intent_features(t) for t in test_df['text']]))

X_train_fused = sparse.hstack([X_train_text, X_train_intent], format='csr')
X_val_fused = sparse.hstack([X_val_text, X_val_intent], format='csr')
X_test_fused = sparse.hstack([X_test_text, X_test_intent], format='csr')

y_train = np.array(train_df['taxonomy'].tolist())
y_val = np.array(val_df['taxonomy'].tolist())
y_test = np.array(test_df['taxonomy'].tolist())

for exp_name, X_tr, X_te in [
    ('Text-only', X_train_text, X_test_text),
    ('Text+Intent Fused', X_train_fused, X_test_fused)
]:
    print(f'=== {exp_name} ===')
    for clf_name, clf in [
        ('LogisticRegression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('CalibratedLinearSVC', CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, dual=False, class_weight='balanced'), cv=3))
    ]:
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        acc = round(accuracy_score(y_test, preds), 4)
        macro_f1 = round(f1_score(y_test, preds, average='macro'), 4)
        weighted_f1 = round(f1_score(y_test, preds, average='weighted'), 4)
        print(f'{clf_name}: Acc={acc}, Macro-F1={macro_f1}, Weighted-F1={weighted_f1}')
        if exp_name == 'Text+Intent Fused' and clf_name == 'CalibratedLinearSVC':
            print('\nClassification Report:')
            print(classification_report(y_test, preds, zero_division=0))
