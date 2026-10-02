---
license: apache-2.0
task_categories:
- text-classification
language:
- hi
- en
tags:
- hinglish
- fraud-detection
- cybercrime
size_categories:
- 10K<n<100K
pretty_name: Indian Scam Communication Dataset
---

# Indian Scam Communication Dataset

## Overview

The Indian Scam Communication Dataset is a curated collection of scam-related messages, call transcripts, and fraud communication patterns commonly observed in India.

The dataset is designed for researchers, students, cybersecurity professionals, NLP engineers, and law-enforcement technology developers working on scam detection, fraud prevention, and conversational AI safety.

## Motivation

India has witnessed a significant rise in:

* Digital Arrest Scams
* Police Impersonation Scams
* Customs and Courier Fraud
* FedEx Parcel Scams
* Money Laundering Threat Scams
* KYC Update Frauds
* Bank Account Verification Scams
* OTP Theft Attempts
* WhatsApp and Telegram Fraud Campaigns

Most publicly available fraud datasets focus on SMS spam and phishing emails. There is a lack of datasets representing modern Indian scam conversations, especially those involving Hinglish and multilingual communication.

## Dataset Contents

The dataset contains approximately 10,000 labeled records.

Possible labels include:

* Scam
* Non-Scam

Communication types include:

* SMS messages
* WhatsApp messages
* Call transcripts
* Chat conversations
* Fraud alerts
* Mixed-language (Hindi-English / Hinglish) content

## Potential Applications

* Scam Message Detection
* Fraud Call Classification
* NLP Research
* Transformer Fine-Tuning
* Cybersecurity Research
* AI Safety Systems
* Educational Projects
* Law Enforcement Analytics

## Example Use Cases

* Fine-tuning MuRIL
* Fine-tuning IndicBERT
* Fine-tuning BERT
* Fine-tuning RoBERTa
* Scam Detection APIs
* Real-Time Call Monitoring Systems
* Fraud Alert Mobile Applications

## Data Fields


| Column   | Description                          |
| -------- | ------------------------------------ |
| text     | Original communication text          |
| label    | Scam or Non-Scam                     |
| source   | SMS, Call, WhatsApp, Telegram, Email |
| language | English, Hindi, Hinglish             |

## Ethical Considerations

All personally identifiable information (PII) should be removed or anonymized before release.

This dataset is intended solely for research, educational, and defensive cybersecurity purposes.

## Citation

If you use this dataset in academic research or commercial products, please cite this repository.
