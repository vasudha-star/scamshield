# ScamShield — PySpark Big Data Pipeline Report (Phase 4)

## 1. Spark Execution & Benchmark Performance
- **Total Records Processed:** 119,340
- **Total Pipeline Time:** 24.89 seconds
- **Processing Throughput:** **4,794.7 records/second**
- **RDD Partitions Executed:** 12
- **Partitioned Write Duration:** 3.47 seconds
- **Output Storage Size:** 74.35 MB (Snappy compressed Parquet)

## 2. Distributed Big Data Analytics

### Ingestion & URL Availability by Source Dataset
| Dataset Source | Records Ingested | Avg Message Length (chars) | Records with URLs |
|---|---|---|---|
| **meajor** | 104,789 | 1218.18 | 62,257 |
| **indian_scam** | 743 | 90.53 | 0 |
| **dravidian_sms** | 3,822 | 144.42 | 1,527 |
| **llm_phishing** | 9,986 | 1222.2 | 4,978 |

### Project Label Class Balance
| Project Label | Distributed Count |
|---|---|
| **benign** | 60,960 |
| **phishing** | 56,782 |
| **spam** | 964 |
| **scam** | 633 |
| **unknown** | 1 |

### Multilingual Language Distribution
| Language Tag | Record Count |
|---|---|
| **en** | 117,913 |
| **hinglish** | 743 |
| **te** | 586 |
| **hi** | 98 |

### Global Text Length Statistics
- **Minimum Character Length:** 1 chars
- **Maximum Character Length:** 30,909 chars
- **Mean Character Length:** 1177.11 chars
- **Standard Deviation:** 1103.03 chars
