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

y_train = np.array(train_df['taxonomy'].tolist())
y_val = np.array(val_df['taxonomy'].tolist())
y_test = np.array(test_df['taxonomy'].tolist())

# Try different TF-IDF configurations
for ngram, sublinear, min_df, max_feat in [
    ((1, 2), True, 1, 5000),
    ((1, 3), True, 1, 10000),
    ((1, 2), True, 2, 3000),
]:
    vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=sublinear, min_df=min_df, max_features=max_feat)
    X_tr = vec.fit_transform(train_df['text'])
    X_val = vec.transform(val_df['text'])
    X_te = vec.transform(test_df['text'])
    
    # Try different C values
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for clf_type in ['LR', 'SVC']:
            if clf_type == 'LR':
                clf = LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=42)
            else:
                clf = LinearSVC(C=C, random_state=42, dual=False, class_weight='balanced')
            clf.fit(X_tr, y_train)
            
            # Val score
            val_preds = clf.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
            
            # Test score
            te_preds = clf.predict(X_te)
            te_acc = accuracy_score(y_test, te_preds)
            te_f1 = f1_score(y_test, te_preds, average='macro', zero_division=0)
            te_wf1 = f1_score(y_test, te_preds, average='weighted', zero_division=0)
            
            if te_f1 > 0.60:
                print(f"ngram={ngram}, C={C}, {clf_type}: Val_F1={val_f1:.4f}, Test_Acc={te_acc:.4f}, Test_MacroF1={te_f1:.4f}, Test_WeightedF1={te_wf1:.4f}")
