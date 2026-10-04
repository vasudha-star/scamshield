# ScamShield — Text Feature Engineering Report (Phase 6)

## 1. Feature Extraction & Vectorizer Specifications
- **Vocabulary Size:** 15,000 n-grams
- **N-gram Range:** Unigrams + Bigrams (1, 2)
- **Sublinear TF Scaling:** Enabled ($1 + \log(\text{tf})$)
- **Train Matrix Dimensions:** 83,537 samples × 15,000 features
- **Matrix Sparsity:** 99.5546%
- **Vectorization Time:** 39.53s (Fit on train)

## 2. Top Discriminative N-grams by Project Label (Train Partition)

### Class: `benign`
| Rank | N-gram Token | Mean TF-IDF Weight |
|---|---|---|
| 1 | `organization` | 0.04152 |
| 2 | `date` | 0.03256 |
| 3 | `product` | 0.02291 |
| 4 | `url` | 0.02288 |
| 5 | `time` | 0.02185 |
| 6 | `thanks` | 0.02048 |
| 7 | `email` | 0.01992 |
| 8 | `phone_number` | 0.01976 |
| 9 | `reference_number` | 0.01895 |
| 10 | `email_address` | 0.01754 |
| 11 | `address` | 0.01719 |
| 12 | `list` | 0.01682 |
| 13 | `file` | 0.01654 |
| 14 | `know` | 0.01586 |
| 15 | `use` | 0.01396 |

### Class: `phishing`
| Rank | N-gram Token | Mean TF-IDF Weight |
|---|---|---|
| 1 | `product` | 0.03187 |
| 2 | `url` | 0.02711 |
| 3 | `financial_info` | 0.02705 |
| 4 | `simbol` | 0.02647 |
| 5 | `simbol simbol` | 0.02416 |
| 6 | `organization` | 0.02263 |
| 7 | `emoji` | 0.0194 |
| 8 | `product product` | 0.01638 |
| 9 | `address` | 0.01612 |
| 10 | `email_separator` | 0.0159 |
| 11 | `emoji emoji` | 0.01388 |
| 12 | `symbol` | 0.01313 |
| 13 | `price` | 0.01255 |
| 14 | `financial_info product` | 0.01226 |
| 15 | `account` | 0.01181 |

### Class: `scam`
| Rank | N-gram Token | Mean TF-IDF Weight |
|---|---|---|
| 1 | `hai` | 0.24852 |
| 2 | `aapke` | 0.17089 |
| 3 | `se` | 0.16926 |
| 4 | `raha` | 0.169 |
| 5 | `raha hoon` | 0.14548 |
| 6 | `bol raha` | 0.14548 |
| 7 | `hoon` | 0.14417 |
| 8 | `bol` | 0.14351 |
| 9 | `se bol` | 0.12288 |
| 10 | `hoon aapke` | 0.07454 |
| 11 | `aapka` | 0.07327 |
| 12 | `cyber` | 0.07056 |
| 13 | `investigation` | 0.061 |
| 14 | `hain` | 0.05927 |
| 15 | `ji` | 0.05757 |

### Class: `spam`
| Rank | N-gram Token | Mean TF-IDF Weight |
|---|---|---|
| 1 | `rs` | 0.1281 |
| 2 | `goo gl` | 0.05932 |
| 3 | `goo` | 0.05893 |
| 4 | `gl` | 0.05729 |
| 5 | `https` | 0.05634 |
| 6 | `http` | 0.05387 |
| 7 | `com` | 0.05008 |
| 8 | `https goo` | 0.04632 |
| 9 | `wallet` | 0.04422 |
| 10 | `vodafone` | 0.04139 |
| 11 | `dial` | 0.03625 |
| 12 | `offer` | 0.03552 |
| 13 | `jio` | 0.03499 |
| 14 | `ly` | 0.03378 |
| 15 | `recharge` | 0.03253 |

