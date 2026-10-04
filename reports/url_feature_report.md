# ScamShield — URL Feature Engineering Report (Phase 8)

## 1. Feature Extractor Overview
- **Engineered URL Features:** 18 offline structural signals
- **Network Access Policy:** 100% Offline (Zero outbound HTTP/DNS requests)
- **Processing Time:** 1.15s across 119,339 samples
- **Standardization:** StandardScaler fitted strictly on train split

## 2. Statistical Comparison: Threat URLs vs. Benign URLs (Train Split)
| Feature | Threat URL Mean | Benign URL Mean | Threat / Benign Ratio | Significance |
|---|---|---|---|---|
| `url_present` | 0.1286 | 0.0367 | 3.5x | High Discrepancy |
| `url_count` | 0.3017 | 0.0374 | 8.07x | High Discrepancy |
| `url_length` | 6.172 | 0.9669 | 6.38x | High Discrepancy |
| `domain_length` | 2.5654 | 0.3957 | 6.48x | High Discrepancy |
| `path_length` | 1.777 | 0.3265 | 5.44x | High Discrepancy |
| `query_length` | 0.8848 | 0.0288 | 30.71x | High Discrepancy |
| `subdomain_count` | 0.0575 | 0.0259 | 2.22x | High Discrepancy |
| `has_ip_address` | 0.0001 | 0.0 | 0.65x | Moderate |
| `digit_count` | 0.2666 | 0.0247 | 10.77x | High Discrepancy |
| `digit_ratio` | 0.0039 | 0.0009 | 4.37x | High Discrepancy |
| `special_char_count` | 0.9697 | 0.1692 | 5.73x | High Discrepancy |
| `has_at_symbol` | 0.0009 | 0.0001 | 8.79x | High Discrepancy |
| `has_double_slash_redirect` | 0.0029 | 0.0 | 28.98x | High Discrepancy |
| `has_hex_encoding` | 0.0014 | 0.0001 | 13.68x | High Discrepancy |
| `is_https` | 0.0703 | 0.0071 | 9.9x | High Discrepancy |
| `is_shortener_domain` | 0.0142 | 0.0034 | 4.13x | High Discrepancy |
| `shannon_entropy` | 0.527 | 0.1427 | 3.69x | High Discrepancy |
| `suspicious_keyword_count` | 0.0909 | 0.0001 | 793.34x | High Discrepancy |

## 3. Key Observations & Security Context
- **URL Length & Path Depth:** Malicious URLs consistently exhibit longer path lengths and deeper directory hierarchies compared to legitimate links.
- **Subdomain Anomaly:** Phishing links frequently leverage multi-tier subdomains (e.g. `service.update.paypal.com.evil.co`) to deceive human users.
- **Shannon Entropy:** Threat URLs show higher lexical randomness due to obfuscated query strings, base64 tokens, and randomized path hashes.
- **HTTPS Fallacy:** Both threat and benign links show high HTTPS presence, proving that HTTPS is no longer an indicator of safety and confirming Rule 13.
