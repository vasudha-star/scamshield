"""Curated Hard-Negative Benchmark Generator for ScamShield (Phase 16).

Creates a gold-standard evaluation set of legitimate, non-malicious transactional
communications that contain high-risk trigger keywords:
('OTP', 'verify', 'urgent', 'account', 'bank', 'password', 'security', 'alert', 'login', 'block').

Taxonomy of Hard Negatives:
1. banking_otp        - Legitimate bank transaction OTPs and debit alerts (SBI, HDFC, ICICI, Axis).
2. security_2fa       - Legitimate authentication and password resets (Google, GitHub, Microsoft).
3. delivery_logistics - Legitimate shipment and delivery notifications (Amazon, Flipkart, BlueDart).
4. statutory_gov      - Legitimate Aadhaar, Income Tax, and DigiLocker notifications.
5. urgent_corporate   - Urgent meeting reminders and operational alerts.
6. multilingual       - Legitimate Hindi, Hinglish, and Telugu authentication messages.

Saves benchmark to: data/gold/hard_negatives_benchmark.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("hard_negatives")

BENCHMARK_RECORDS: list[dict[str, str]] = [
    # --- 1. Banking & UPI Alerts ---
    {
        "record_id": "HN_BANK_01",
        "category": "banking_otp",
        "language": "en",
        "text": "Your SBI NetBanking login OTP is 582914. Valid for 10 minutes. Never share your OTP, password or CVV with anyone including bank staff.",
        "trigger_keywords": "otp, bank, password",
    },
    {
        "record_id": "HN_BANK_02",
        "category": "banking_otp",
        "language": "en",
        "text": "Alert: Rs 1,450.00 debited from HDFC Bank Account **4910 on 03-Oct-26 via UPI. If not done by you, call 18002583838 immediately to block card.",
        "trigger_keywords": "alert, bank, account, block, immediately",
    },
    {
        "record_id": "HN_BANK_03",
        "category": "banking_otp",
        "language": "en",
        "text": "ICICI Bank: OTP for online transaction of INR 3,299.00 at Flipkart is 839102. Do not share OTP with anyone to maintain security.",
        "trigger_keywords": "bank, otp, security",
    },
    {
        "record_id": "HN_BANK_04",
        "category": "banking_otp",
        "language": "en",
        "text": "Axis Bank Security Alert: Your credit card payment of INR 8,450 is due on 05-Oct. Please pay through official Axis Mobile App or netbanking.",
        "trigger_keywords": "bank, security, alert",
    },
    {
        "record_id": "HN_BANK_05",
        "category": "banking_otp",
        "language": "en",
        "text": "Your monthly bank account statement for account ending in 9812 has been generated. Login to netbanking portal https://netbanking.hdfcbank.com to view.",
        "trigger_keywords": "bank, account, login",
    },
    {
        "record_id": "HN_BANK_06",
        "category": "banking_otp",
        "language": "en",
        "text": "Urgent: Unusual sign-in attempt detected on your bank netbanking. If this was you, please verify via our official mobile app.",
        "trigger_keywords": "urgent, bank, verify",
    },
    {
        "record_id": "HN_BANK_07",
        "category": "banking_otp",
        "language": "en",
        "text": "Dear Customer, INR 5,000.00 withdrawn from ATM using SBI Debit Card ending in 1102. Call bank toll-free if unauthorized.",
        "trigger_keywords": "bank",
    },
    {
        "record_id": "HN_BANK_08",
        "category": "banking_otp",
        "language": "en",
        "text": "Kotak Mahindra Bank: Verify your updated residential address for credit card delivery by visiting your nearest branch with Aadhaar.",
        "trigger_keywords": "bank, verify",
    },
    {
        "record_id": "HN_BANK_09",
        "category": "banking_otp",
        "language": "en",
        "text": "Punjab National Bank: Account credited with salary INR 65,000. Available balance in your savings account is INR 82,410.",
        "trigger_keywords": "bank, account",
    },
    {
        "record_id": "HN_BANK_10",
        "category": "banking_otp",
        "language": "en",
        "text": "Bank alert: Your debit card will expire this month. A replacement card has been dispatched to your registered communication address.",
        "trigger_keywords": "bank, alert",
    },

    # --- 2. Security & 2FA Authentication ---
    {
        "record_id": "HN_SEC_01",
        "category": "security_2fa",
        "language": "en",
        "text": "Google verification code: G-739201. Use this code to verify your Google Account. Never share this code with anyone.",
        "trigger_keywords": "verify, account",
    },
    {
        "record_id": "HN_SEC_02",
        "category": "security_2fa",
        "language": "en",
        "text": "GitHub: A new personal access token was generated on your account. If you did not create this token, review security settings at https://github.com/settings/security.",
        "trigger_keywords": "account, security",
    },
    {
        "record_id": "HN_SEC_03",
        "category": "security_2fa",
        "language": "en",
        "text": "Microsoft Security Alert: A new sign-in from Firefox on Windows was detected for your account. If this was not you, reset your password.",
        "trigger_keywords": "security, alert, account, password",
    },
    {
        "record_id": "HN_SEC_04",
        "category": "security_2fa",
        "language": "en",
        "text": "Apple ID: Your Apple Account password was recently changed. If you made this change, you can safely disregard this email.",
        "trigger_keywords": "account, password",
    },
    {
        "record_id": "HN_SEC_05",
        "category": "security_2fa",
        "language": "en",
        "text": "Your Uber login code is 8492. Reply STOP to unsubscribe. Never share your login code with anyone including Uber drivers.",
        "trigger_keywords": "login",
    },
    {
        "record_id": "HN_SEC_06",
        "category": "security_2fa",
        "language": "en",
        "text": "WhatsApp code 918-204. You can also tap this link to verify your phone number: https://v.whatsapp.com/918204. Do not share this code.",
        "trigger_keywords": "verify",
    },
    {
        "record_id": "HN_SEC_07",
        "category": "security_2fa",
        "language": "en",
        "text": "LinkedIn Security: Someone requested a password reset for your account. Click here if you requested it: https://linkedin.com/checkpoint/rp.",
        "trigger_keywords": "security, password, account",
    },
    {
        "record_id": "HN_SEC_08",
        "category": "security_2fa",
        "language": "en",
        "text": "Slack notification: You have been invited to join Workspace 'ScamShield Engineering'. Please verify your email to accept the invite.",
        "trigger_keywords": "verify",
    },
    {
        "record_id": "HN_SEC_09",
        "category": "security_2fa",
        "language": "en",
        "text": "AWS Security Hub: Monthly compliance report for your account 8391-2831 is now ready in the AWS Management Console.",
        "trigger_keywords": "security, account",
    },
    {
        "record_id": "HN_SEC_10",
        "category": "security_2fa",
        "language": "en",
        "text": "Urgent password update required: Your enterprise single sign-on password will expire in 3 days as per company IT security policy.",
        "trigger_keywords": "urgent, password, security",
    },

    # --- 3. Delivery & Logistics ---
    {
        "record_id": "HN_DELIV_01",
        "category": "delivery_logistics",
        "language": "en",
        "text": "Amazon Delivery Update: Your package with order #402-819283-1928 is out for delivery today with courier delivery associate Ramesh.",
        "trigger_keywords": "update",
    },
    {
        "record_id": "HN_DELIV_02",
        "category": "delivery_logistics",
        "language": "en",
        "text": "Flipkart: Your order has been delivered. Share delivery verification PIN 4920 with delivery agent only when you receive the package.",
        "trigger_keywords": "verify",
    },
    {
        "record_id": "HN_DELIV_03",
        "category": "delivery_logistics",
        "language": "en",
        "text": "BlueDart Alert: Shipment with waybill 84920194 is expected to arrive by 6 PM. Track status on official site https://bluedart.com.",
        "trigger_keywords": "alert",
    },
    {
        "record_id": "HN_DELIV_04",
        "category": "delivery_logistics",
        "language": "en",
        "text": "Zomato: Delivery partner is arriving with your order. Provide delivery confirmation code 81 to verify receipt of food.",
        "trigger_keywords": "verify",
    },
    {
        "record_id": "HN_DELIV_05",
        "category": "delivery_logistics",
        "language": "en",
        "text": "Urgent update on shipment: Delivery attempted for package #91823. Recipient unavailable. Rescheduled for tomorrow morning.",
        "trigger_keywords": "urgent, update",
    },

    # --- 4. Statutory & Government Services ---
    {
        "record_id": "HN_GOV_01",
        "category": "statutory_gov",
        "language": "en",
        "text": "UIDAI: Aadhaar OTP is 391048 for generating e-Aadhaar. Valid for 10 minutes. Do not share OTP with unauthorized callers.",
        "trigger_keywords": "otp",
    },
    {
        "record_id": "HN_GOV_02",
        "category": "statutory_gov",
        "language": "en",
        "text": "Income Tax Department: e-Filing confirmation. ITR-1 for AY 2026-27 has been successfully e-verified using Aadhaar OTP.",
        "trigger_keywords": "verify, otp",
    },
    {
        "record_id": "HN_GOV_03",
        "category": "statutory_gov",
        "language": "en",
        "text": "Passport Seva Alert: Your appointment at PSK Bengaluru is scheduled for 10-Oct-26 at 09:30 AM. Bring original documents to verify.",
        "trigger_keywords": "alert, verify",
    },
    {
        "record_id": "HN_GOV_04",
        "category": "statutory_gov",
        "language": "en",
        "text": "DigiLocker Security Alert: Your Driving License document was successfully fetched into your DigiLocker account.",
        "trigger_keywords": "security, alert, account",
    },
    {
        "record_id": "HN_GOV_05",
        "category": "statutory_gov",
        "language": "en",
        "text": "EPFO India: Passbook update for UAN 1009281928. Employer contribution of Rs 1,800 has been credited to your provident fund account.",
        "trigger_keywords": "update, account",
    },

    # --- 5. Multilingual Hard Negatives (Hindi / Hinglish / Telugu) ---
    {
        "record_id": "HN_MULTI_01",
        "category": "multilingual",
        "language": "hi",
        "text": "भारतीय स्टेट बैंक: आपके खाते से रुपये 500 का यूपीआई भुगतान सफल रहा। किसी भी समस्या के लिए 1800112211 पर संपर्क करें।",
        "trigger_keywords": "bank, account",
    },
    {
        "record_id": "HN_MULTI_02",
        "category": "multilingual",
        "language": "hinglish",
        "text": "Aapka SBI netbanking login OTP hai 492018. Yeh OTP 10 minute ke liye valid hai. Kripya kisi ke saath share na karein.",
        "trigger_keywords": "otp, bank, login",
    },
    {
        "record_id": "HN_MULTI_03",
        "category": "multilingual",
        "language": "hinglish",
        "text": "HDFC Bank Alert: Aapke account se Rs 2,000 debit hua hai. Agar aapne yeh transaction nahi kiya hai toh turant card block karein.",
        "trigger_keywords": "bank, alert, account, block",
    },
    {
        "record_id": "HN_MULTI_04",
        "category": "multilingual",
        "language": "hinglish",
        "text": "Urgent reminder: Kal office meeting 10 baje hai, please sabhi presentations verify karke ready rakhein.",
        "trigger_keywords": "urgent, verify",
    },
    {
        "record_id": "HN_MULTI_05",
        "category": "multilingual",
        "language": "hinglish",
        "text": "Flipkart order deliver ho chuka hai. Delivery agent ko rating dene ke liye app login karein.",
        "trigger_keywords": "login",
    },
    {
        "record_id": "HN_MULTI_06",
        "category": "multilingual",
        "language": "te",
        "text": "Mee HDFC Bank account nunchi Rs. 500 debit ayindi. Ee transaction meeru cheyakapothe bank branch ki alert cheyandi.",
        "trigger_keywords": "bank, account, alert",
    },
    {
        "record_id": "HN_MULTI_07",
        "category": "multilingual",
        "language": "te",
        "text": "SBI Netbanking OTP: 839102. Ee OTP 5 nimishalu matrame pani chestundi. Evaritho share cheyavaddu.",
        "trigger_keywords": "otp, bank",
    },
    {
        "record_id": "HN_MULTI_08",
        "category": "multilingual",
        "language": "te",
        "text": "Urgent: Repu morning doctor appointment 9 gantalaku confirm ayindi. Time ki raavalsindi ga manavi.",
        "trigger_keywords": "urgent",
    },
    {
        "record_id": "HN_MULTI_09",
        "category": "multilingual",
        "language": "te",
        "text": "Mee Amazon delivery package repu dispatch avthundi. Address verify cheyandi.",
        "trigger_keywords": "verify",
    },
    {
        "record_id": "HN_MULTI_10",
        "category": "multilingual",
        "language": "te",
        "text": "Bank security alert: Mee debit card PIN change successful ga jarigindi.",
        "trigger_keywords": "bank, security, alert",
    },
]


def create_hard_negatives_benchmark() -> pd.DataFrame:
    """Creates and saves the curated hard-negative benchmark dataset."""
    out_dir = PROJECT_ROOT / "data" / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hard_negatives_benchmark.csv"

    df = pd.DataFrame(BENCHMARK_RECORDS)
    df["ground_truth"] = 0  # All are strictly legitimate communications
    df["project_label"] = "benign"

    df.to_csv(out_file, index=False, encoding="utf-8")
    logger.info(f"Saved {len(df)} curated hard-negative benchmark records to {out_file}.")
    return df


if __name__ == "__main__":
    create_hard_negatives_benchmark()
