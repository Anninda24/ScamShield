import os
import random
import re
import pandas as pd

from synth import (
    _build_slots,
    _gen_single_templates,
    _gen_multiturn_templates,
    MFS_BRANDS,
    BANK_BRANDS,
    COURIER_BRANDS,
    TELCO_BRANDS,
    ECOM_BRANDS,
    EMPLOYER_BRANDS,
    CRYPTO_BRANDS,
    PERSON_NAMES,
    PHISH_DOMAINS,
    LEGIT_DOMAINS,
)
from noise import apply_noise

# Map v1 family_ids to scenario_id
V1_FAMILY_TO_SCENARIO = {
    "c0_otp_warn": "0_otp_genuine_notice",
    "c0_tx_confirm": "0_tx_confirmation",
    "c0_parcel_legit": "0_delivery_genuine",
    "c0_mt_delivery": "0_delivery_genuine",
    "c0_mt_delivery_mix": "0_delivery_genuine",
    "c0_sec_alert": "0_security_warning",
    "c0_mt_support": "0_security_warning",
    "c0_bill_remind": "0_bill_reminder",
    "c0_cashback_real": "0_cashback_genuine",
    "c0_personal": "0_personal_chat",
    "c0_mt_chat": "0_personal_chat",
    "c1_kyc_lock": "1_account_lock_kyc_pin",
    "c1_mt_bank": "1_account_lock_kyc_pin",
    "c1_mt_agent_bn": "1_account_lock_kyc_pin",
    "c1_mistake_tx": "1_mistaken_transfer",
    "c1_mt_agent": "1_mistaken_transfer",
    "c1_card_refund": "1_card_refund_cvv",
    "c2_lottery_link": "2_lottery_cashback_link",
    "c2_mt_lottery_bn": "2_lottery_cashback_link",
    "c2_prize_fee": "2_prize_advance_fee",
    "c2_mt_prize": "2_prize_advance_fee",
    "c2_prize_app": "2_prize_spin_apk",
    "c2_prize_details": "2_prize_details_reply",
    "c3_task_earn": "3_task_earning",
    "c3_mt_task": "3_task_earning",
    "c3_invest_fee": "3_investment_job_fee",
    "c3_mt_jobfee": "3_investment_job_fee",
    "c3_mt_long": "3_investment_job_fee",
    "c3_crypto": "3_crypto_vip_signals",
    "c3_abroad_job": "3_abroad_visa_fee",
    "c4_parcel_fee": "4_parcel_storage_fee",
    "c4_customs": "4_parcel_customs_fee",
    "c4_mt_customs": "4_parcel_customs_fee",
    "c4_mt_customs_bn": "4_parcel_customs_fee",
    "c4_addr_link": "4_parcel_address_link",
    "c4_mt_addr_mix": "4_parcel_address_link",
    "c4_addr_call": "4_parcel_address_call",
    "c5_promo_offer": "5_promo_sales_offer",
    "c5_mt_promo": "5_promo_sales_offer",
    "c5_mt_promo_bn": "5_promo_sales_offer",
    "c5_promo_app": "5_promo_app_install",
    "c5_promo_biz": "5_promo_real_estate",
    "c5_promo_course": "5_promo_course_ad",
}


def _wrap_multiturn(rng: random.Random, s: dict, body_turns: list, lead_in_style: str = "neutral"):
    spk_s, spk_r = rng.choice([("S", "R"), ("Caller", "You"), ("Agent", "Customer"), ("A", "B")])
    if lead_in_style == "greet":
        intro = [
            f"{spk_s}: Assalamu alaikum {s['name']}, kemon achen?",
            f"{spk_r}: Wa alaikumussalam, bhalo. Ke bolchen?",
        ]
    elif lead_in_style == "casual":
        intro = [
            f"{spk_s}: Hello, are you free for a moment?",
            f"{spk_r}: Yes, who is this?",
        ]
    elif lead_in_style == "bangla":
        intro = [
            f"{spk_s}: হ্যালো, একটু কথা বলা যাবে?",
            f"{spk_r}: জ্বী বলুন, কে বলছেন?",
        ]
    else:
        intro = []
    formatted = intro + [f"{spk_s if i % 2 == 0 else spk_r}: {t}" for i, t in enumerate(body_turns)]
    return len(formatted), " ||| ".join(formatted)


def _gen_new_scenarios(rng: random.Random, s: dict) -> list:
    """Return list of (scenario_id, family_id, language, label, techniques, action, n_turns, is_hard_negative, text)."""
    rows = []

    def add(scen, lang, lbl, tech, act, hard, txt, n_t=1):
        rows.append((scen, scen, lang, lbl, tech, act, n_t, hard, txt))

    # =========================================================================
    # CLASS 0: 14 NEW SCENARIOS (+ 7 v1 = 21 scenarios total)
    # =========================================================================
    # 0_work_chat
    add("0_work_chat", "english", 0, "NONE", 0, False,
        f"Team update: Sprint review moved to 4 PM by {s['name']}. Please push your branch before then.")
    add("0_work_chat", "banglish", 0, "NONE", 0, False,
        f"{s['name']}, client er presentation deck ta ajke dupurer moddhe mail kore dio.")
    add("0_work_chat", "bangla", 0, "NONE", 0, False,
        f"{s['name']}, অফিসের রিপোর্টটা রেডি হলে গুগল ড্রাইভে আপলোড করে দিও।")
    n_t, mt = _wrap_multiturn(rng, s, [
        "Did you check the server logs from last night?",
        "Yes, latency spiked around 2 AM, already patched.",
        "Great, let's monitor for an hour."
    ], "casual")
    add("0_work_chat", "mixed", 0, "NONE", 0, False, mt, n_t)

    # 0_appointment_reminder
    add("0_appointment_reminder", "english", 0, "NONE", 0, False,
        f"Clinic Reminder: Your dental checkup with Dr. {s['name']} is tomorrow at 11:30 AM. Reply C to confirm.")
    add("0_appointment_reminder", "bangla", 0, "NONE", 0, False,
        f"হাসপাতাল নোটিশ: আগামীকাল সকাল ১০টায় আপনার অ্যাপয়েন্টমেন্ট নির্ধারিত আছে।")
    add("0_appointment_reminder", "banglish", 0, "NONE", 0, False,
        f"Diagnostic center: Apnar blood test report ready. Kal bikal 5 tay collect korte parben.")
    add("0_appointment_reminder", "mixed", 0, "NONE", 0, False,
        f"Passport office appointment confirmed for 12 Dec at 9 AM. Bring original NID.")

    # 0_bank_statement
    add("0_bank_statement", "english", 0, "NONE", 0, True,
        f"{s['bank']}: Monthly e-statement for A/C XX{s['otp'][:4]} is ready. Closing balance BDT {s['bal']}. View in official app.")
    add("0_bank_statement", "bangla", 0, "NONE", 0, True,
        f"{s['bank']}: আপনার মাসিক ব্যাংক স্টেটমেন্ট তৈরি হয়েছে। ব্যালেন্স {s['bal']} টাকা।")
    add("0_bank_statement", "banglish", 0, "NONE", 0, True,
        f"{s['bank']} card statement generated. Total due Tk {s['amt']}, min due Tk 500. Pay via app.")
    add("0_bank_statement", "mixed", 0, "NONE", 0, True,
        f"{s['bank']}: Your FDR interest of Tk {s['amt']} has been credited. Avl bal Tk {s['bal']}.")

    # 0_service_notice_link (hard legit with link)
    add("0_service_notice_link", "english", 0, "NONE", 0, True,
        f"{s['telco']} Notice: Scheduled maintenance tonight 2 AM - 4 AM. Status updates: {s['lurl']}")
    add("0_service_notice_link", "bangla", 0, "NONE", 0, True,
        f"{s['mfs']} সেবা বিজ্ঞপ্তি: আজ রাত ২টা থেকে ৪টা পর্যন্ত অ্যাপ রক্ষণাবেক্ষণ চলবে। বিস্তারিত: {s['lurl']}")
    add("0_service_notice_link", "banglish", 0, "NONE", 0, True,
        f"DESCO bill copy download korte parben ekhan theke: {s['lurl']}. Kono fee lagbe na.")
    add("0_service_notice_link", "mixed", 0, "NONE", 0, True,
        f"{s['ecom']} privacy policy updated. Read details at {s['lurl']}.")

    # 0_family_transfer_chat
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Ammu, bazar er jonno Tk {s['amt']} pathalam {s['mfs']} e.",
        "Accha baba, check korchi. Balance asche.",
        "Thik ase, lagle janio."
    ], "greet")
    add("0_family_transfer_chat", "banglish", 0, "NONE", 0, True, mt, n_t)
    add("0_family_transfer_chat", "bangla", 0, "NONE", 0, True,
        f"আব্বু, তোমার {s['mfs']} নম্বরে {s['amt']} টাকা পাঠিয়েছি, পেয়েছো কিনা জানিও।")
    add("0_family_transfer_chat", "english", 0, "NONE", 0, True,
        f"Hey sis, I sent Tk {s['amt']} to your {s['mfs']} for mom's medicine. Let me know when you get it.")
    add("0_family_transfer_chat", "mixed", 0, "NONE", 0, True,
        f"Bhai, dinner bill er {s['amt']} taka tor {s['mfs']} e diye disi.")

    # 0_kyc_branch_notice (hard negative twin of 1_kyc_lock)
    add("0_kyc_branch_notice", "english", 0, "NONE", 0, True,
        f"{s['bank']}: Please update your KYC by visiting your nearest branch with original NID. Never share PIN or OTP via phone.")
    add("0_kyc_branch_notice", "bangla", 0, "NONE", 0, True,
        f"{s['bank']}: আপনার কেওয়াইসি হালনাগাদ করতে নিকটস্থ শাখায় এনআইডি নিয়ে আসুন। ফোনে পিন বা ওটিপি দেবেন না।")
    add("0_kyc_branch_notice", "banglish", 0, "NONE", 0, True,
        f"{s['mfs']} notice: KYC verify korte shudhu official app use korun. Agent ke PIN diben na.")
    add("0_kyc_branch_notice", "mixed", 0, "NONE", 0, True,
        f"SIM biometric re-verification: visit {s['telco']} customer care center with NID. No link or fee required.")

    # 0_self_pwd_reset (hard negative twin of 1_pwd_reset_link)
    add("0_self_pwd_reset", "english", 0, "NONE", 0, True,
        f"You requested a password reset on {s['ecom']}. If you did not request this, ignore this message. Do not share your code.")
    add("0_self_pwd_reset", "banglish", 0, "NONE", 0, True,
        f"{s['mfs']} PIN reset request peyeche. Apni na korle 16247 e call korun. Code kauke bolben na.")
    add("0_self_pwd_reset", "bangla", 0, "NONE", 0, True,
        f"আপনি পাসওয়ার্ড পরিবর্তনের অনুরোধ করেছেন। না করে থাকলে এটি উপেক্ষা করুন, কোড শেয়ার করবেন না।")
    add("0_self_pwd_reset", "mixed", 0, "NONE", 0, True,
        f"Login alert: {s['bank']} app password changed successfully. Contact branch if unauthorized.")

    # 0_cod_delivery_legit (hard negative twin of 4_cod_confirm)
    add("0_cod_delivery_legit", "english", 0, "NONE", 0, True,
        f"{s['ecom']} Rider {s['name']} is on the way with order #{s['parcel']}. Pay Tk {s['amt']} cash on delivery upon receiving.")
    add("0_cod_delivery_legit", "bangla", 0, "NONE", 0, True,
        f"আপনার অর্ডার #{s['parcel']} ডেলিভারির জন্য বের হয়েছে। পণ্য হাতে পেয়ে {s['amt']} টাকা ক্যাশ দেবেন।")
    add("0_cod_delivery_legit", "banglish", 0, "NONE", 0, True,
        f"{s['courier']}: Rider {s['name']} apnar parcel niye asche. Cash on delivery Tk {s['amt']}.")
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Sir, ami {s['courier']} rider bolchi, apnar parcel niye gate e aschi.",
        "Accha ashen, cash on delivery koto?",
        f"Tk {s['amt']} sir, packet check kore taka diyen."
    ], "neutral")
    add("0_cod_delivery_legit", "mixed", 0, "NONE", 0, True, mt, n_t)

    # 0_emi_reminder
    add("0_emi_reminder", "english", 0, "NONE", 0, True,
        f"{s['bank']}: Reminder that your auto-debit EMI of Tk {s['amt']} is scheduled on 5th of this month.")
    add("0_emi_reminder", "bangla", 0, "NONE", 0, True,
        f"{s['bank']}: আপনার মাসিক কিস্তি {s['amt']} টাকা আগামী ৫ তারিখ স্বয়ংক্রিয়ভাবে কাটা হবে।")
    add("0_emi_reminder", "banglish", 0, "NONE", 0, True,
        f"{s['bank']} loan EMI Tk {s['amt']} due soon. Account e balance rakhun.")
    add("0_emi_reminder", "mixed", 0, "NONE", 0, True,
        f"Credit card bill payment of Tk {s['amt']} received by {s['bank']}. Thank you.")

    # 0_govt_portal_legit
    add("0_govt_portal_legit", "english", 0, "NONE", 0, False,
        "NID Service: Your smart NID card is ready for collection at your Upazila Election Office.")
    add("0_govt_portal_legit", "bangla", 0, "NONE", 0, False,
        "নির্বাচন কমিশন: আপনার স্মার্ট এনআইডি কার্ড উপজেলা অফিসে এসেছে। মূল স্লিপসহ সংগ্রহ করুন।")
    add("0_govt_portal_legit", "banglish", 0, "NONE", 0, False,
        "BRTA notice: Apnar driving license card print complete. Office theke collect korun.")
    add("0_govt_portal_legit", "mixed", 0, "NONE", 0, False,
        "NBR e-TIN confirmation: Your tax return acknowledgement slip is available in portal.")

    # 0_telco_pack_expiry
    add("0_telco_pack_expiry", "english", 0, "NONE", 0, True,
        f"{s['telco']}: Your 5GB internet pack expires tonight at 11:59 PM. Dial *121# or open app to renew.")
    add("0_telco_pack_expiry", "bangla", 0, "NONE", 0, True,
        f"{s['telco']}: আপনার ইন্টারনেট প্যাকের মেয়াদ আজ শেষ হবে। রিনিউ করতে অফিসিয়াল অ্যাপ দেখুন।")
    add("0_telco_pack_expiry", "banglish", 0, "NONE", 0, True,
        f"Dear customer, apnar {s['telco']} minute bundle sesh hoye jabe ajke. Dial *123*1#.")
    add("0_telco_pack_expiry", "mixed", 0, "NONE", 0, True,
        f"{s['telco']} balance low: remaining Tk 4.20. Recharge via {s['mfs']} or official app.")

    # 0_subscription_renewal
    add("0_subscription_renewal", "english", 0, "NONE", 0, False,
        f"Your cloud storage subscription (Tk {s['amt']}/month) has been renewed. Receipt sent to email.")
    add("0_subscription_renewal", "bangla", 0, "NONE", 0, False,
        f"আপনার মাসিক নিউজ অ্যাপ সাবস্ক্রিপশন সফলভাবে নবায়ন হয়েছে।")
    add("0_subscription_renewal", "banglish", 0, "NONE", 0, False,
        f"WiFi bill Tk {s['amt']} paid to {s['telco']}. Connection active for next 30 days.")
    add("0_subscription_renewal", "mixed", 0, "NONE", 0, False,
        f"Spotify / OTT monthly plan renewed for Tk {s['amt']}. Manage in account settings.")

    # 0_interview_legit (hard negative twin of 3_data_entry_job)
    add("0_interview_legit", "english", 0, "NONE", 0, True,
        f"HR, {s['emp']}: Your interview is scheduled Tuesday 10 AM at our Gulshan office. Bring CV. No fee required.")
    add("0_interview_legit", "bangla", 0, "NONE", 0, True,
        f"{s['emp']} এইচআর: আগামী মঙ্গলবার সকাল ১০টায় আপনার চাকরির সাক্ষাৎকার। সিভি সাথে আনবেন।")
    add("0_interview_legit", "banglish", 0, "NONE", 0, True,
        f"{s['emp']} office theke bolchi: apnar CV shortlist hoyeche, Thursday 11 AM e interview.")
    add("0_interview_legit", "mixed", 0, "NONE", 0, True,
        f"Walk-in interview at {s['emp']} Dhaka HQ on Sunday. Bring NID copy and certificates.")

    # 0_exam_university
    add("0_exam_university", "english", 0, "NONE", 0, False,
        "University Registrar: Final exam routine for Fall semester has been published on the student portal.")
    add("0_exam_university", "bangla", 0, "NONE", 0, False,
        "ক্লাস নোটিশ: আগামীকাল সকাল ৯টার ক্লাস ১০টায় অনুষ্ঠিত হবে।")
    add("0_exam_university", "banglish", 0, "NONE", 0, False,
        f"Bhai {s['name']}, kal er lab report ar assignment er deadline ki Friday?")
    add("0_exam_university", "mixed", 0, "NONE", 0, False,
        "Library notice: The book you reserved is ready for pickup within 2 days.")

    # =========================================================================
    # CLASS 1: 13 NEW SCENARIOS (+ 3 v1 = 16 scenarios total)
    # =========================================================================
    # 1_remote_access_support
    add("1_remote_access_support", "english", 1, "T1|T2|T5", 4, True,
        f"{s['mfs']} Tech Support: To fix your failed transaction, install AnyDesk and share the 9-digit code plus your PIN.")
    add("1_remote_access_support", "bangla", 1, "T1|T2|T5", 4, True,
        f"{s['bank']} সাপোর্ট: আপনার অ্যাপ ঠিক করতে টিমভিউয়ার ইনস্টল করে কোড ও পিন দিন।")
    add("1_remote_access_support", "banglish", 1, "T1|T2|T5", 4, True,
        f"{s['mfs']} helpline: Screen share app download korun ar code ta bolen, na hole taka ferot paben na.")
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Sir, ami {s['mfs']} technical team theke bolchi. Apnar app e bug hoyeche.",
        "Ki bug? Amar balance to thik ache.",
        "Security issue ache. Play Store theke AnyDesk install kore code ta den ar PIN verify korun."
    ], "greet")
    add("1_remote_access_support", "mixed", 1, "T1|T2|T5", 4, True, mt, n_t)

    # 1_wallet_limit_upgrade
    add("1_wallet_limit_upgrade", "english", 1, "T1|T2", 1, True,
        f"{s['mfs']} Notice: Upgrade your daily wallet limit to Tk 1,00,000. Reply with your PIN and current OTP.")
    add("1_wallet_limit_upgrade", "bangla", 1, "T1|T2", 1, True,
        f"{s['mfs']}: আপনার লেনদেনের সীমা বাড়াতে এখনই পিন ও এনআইডি নম্বর পাঠান।")
    add("1_wallet_limit_upgrade", "banglish", 1, "T1|T2|T5", 1, True,
        f"SABDHAN! {s['mfs']}: APNAR WALLET LIMIT BARANO HOBE. CARD NUMBER, EXPIRY AR PIN DEN.")
    add("1_wallet_limit_upgrade", "mixed", 1, "T1|T2", 1, True,
        f"VIP {s['mfs']} limit upgrade: verify with 6-digit OTP to unlock Tk 50,000 daily transfer.")

    # 1_sim_kyc_nid_photo
    add("1_sim_kyc_nid_photo", "english", 1, "T1|T2|T5", 1, True,
        f"{s['telco']} Alert: Your SIM registration is incomplete. Send NID front/back photo and MFS PIN within 2h or SIM stops.")
    add("1_sim_kyc_nid_photo", "bangla", 1, "T1|T2|T5", 1, True,
        f"{s['telco']} সতর্কতা: ২ ঘণ্টার মধ্যে এনআইডি ছবি ও {s['mfs']} পিন না দিলে সিম বন্ধ হয়ে যাবে।")
    add("1_sim_kyc_nid_photo", "banglish", 1, "T1|T2|T5", 1, True,
        f"Apnar SIM ar {s['mfs']} bondho hoye jabe. NID front back chobi ar OTP pathan ekhoni.")
    add("1_sim_kyc_nid_photo", "mixed", 1, "T1|T2|T5", 1, True,
        f"Biometric mismatch on {s['telco']}! Reply with NID number and {s['mfs']} PIN to avoid deactivation.")

    # 1_pwd_reset_link
    add("1_pwd_reset_link", "english", 1, "T1|T2|T5", 3, True,
        f"{s['bank']} Alert: Unauthorized password change detected! Cancel immediately by entering PIN at {s['purl']}")
    add("1_pwd_reset_link", "bangla", 1, "T1|T2|T5", 3, True,
        f"{s['mfs']} নিরাপত্তা সতর্কতা: আপনার পাসওয়ার্ড হ্যাক হওয়ার ঝুঁকিতে। পিন দিয়ে যাচাই করুন: {s['purl']}")
    add("1_pwd_reset_link", "banglish", 1, "T1|T2|T5", 3, True,
        f"Apnar {s['bank']} password reset korte {s['purl']} e dhuke old PIN ar OTP din.")
    add("1_pwd_reset_link", "mixed", 1, "T1|T2", 3, True,
        f"{s['mfs']} web login: confirm your PIN and OTP on {s['purl']} to restore access.")

    # 1_agent_commission_fraud
    add("1_agent_commission_fraud", "english", 1, "T1|T2|T4", 1, True,
        f"{s['mfs']} Agent Desk: Claim Tk {s['amt']} monthly agent commission by sharing your merchant PIN and OTP.")
    add("1_agent_commission_fraud", "bangla", 1, "T1|T2|T4", 1, True,
        f"{s['mfs']} এজেন্ট কমিশন: {s['amt']} টাকা বোনাস পেতে আপনার এজেন্ট পিন ও ওটিপি দিন।")
    add("1_agent_commission_fraud", "banglish", 1, "T1|T2|T4", 1, True,
        f"Agent bhai, apnar {s['mfs']} commission {s['amt']} taka add korte OTP ta bolen.")
    add("1_agent_commission_fraud", "mixed", 1, "T1|T2|T4", 1, True,
        f"Merchant cashback Tk {s['amt']} ready for {s['mfs']}! Enter PIN at {s['purl']}.")

    # 1_reply_code_alert
    add("1_reply_code_alert", "english", 1, "T1|T2|T5", 1, True,
        f"URGENT {s['mfs']}: Reply to this SMS with the 6-digit verification code you received to stop a Tk {s['amt']} debit.")
    add("1_reply_code_alert", "bangla", 1, "T1|T2|T5", 1, True,
        f"জরুরি: আপনার {s['mfs']} থেকে {s['amt']} টাকা কাটা ঠেকাতে এখনই আসা ৬ সংখ্যার কোডটি রিপ্লাই করুন।")
    add("1_reply_code_alert", "banglish", 1, "T1|T2|T5", 1, True,
        f"Apnar number e ekta 6 digit code gese. Account safe rakhte code ta reply korun ekhoni.")
    add("1_reply_code_alert", "mixed", 1, "T1|T2|T5", 1, True,
        f"Security code {s['otp']}? Reply the code here to verify your {s['bank']} login.")

    # 1_dormant_reactivation
    add("1_dormant_reactivation", "english", 1, "T1|T2|T5", 1, True,
        f"{s['bank']}: Your account is marked dormant. Reactivate by sending A/C number, ATM PIN, and OTP.")
    add("1_dormant_reactivation", "bangla", 1, "T1|T2|T5", 1, True,
        f"{s['bank']}: আপনার অ্যাকাউন্ট নিষ্ক্রিয় হয়েছে। চালু করতে এটিএম পিন ও ওটিপি পাঠান।")
    add("1_dormant_reactivation", "banglish", 1, "T1|T2|T5", 1, True,
        f"Apnar {s['mfs']} account dormant hoye gese. Chalu korte PIN ar NID number pathan.")
    add("1_dormant_reactivation", "mixed", 1, "T1|T2", 1, True,
        f"Reactivate {s['mfs']} wallet: share last 4 digits of NID and your 5-digit PIN.")

    # 1_card_unblock_otp
    add("1_card_unblock_otp", "english", 1, "T1|T2|T5", 1, True,
        f"{s['bank']} Card Dept: Your debit card is temporarily blocked. Share OTP and CVV to unblock.")
    add("1_card_unblock_otp", "bangla", 1, "T1|T2|T5", 1, True,
        f"{s['bank']}: আপনার ডেবিট কার্ড ব্লক করা হয়েছে। আনব্লক করতে সিভিভি ও ওটিপি দিন।")
    add("1_card_unblock_otp", "banglish", 1, "T1|T2|T5", 1, True,
        f"Visa card block hoyeche. Unblock korte card expiry ar OTP ta bolen.")
    add("1_card_unblock_otp", "mixed", 1, "T1|T2|T5", 1, True,
        f"{s['bank']} card alert: unblock within 30 min by replying OTP.")

    # 1_govt_subsidy_pin
    add("1_govt_subsidy_pin", "english", 1, "T1|T2|T4", 1, False,
        f"Govt Relief Fund: Receive Tk {s['amt']} directly in {s['mfs']}. Share your {s['mfs']} PIN and OTP to verify wallet.")
    add("1_govt_subsidy_pin", "bangla", 1, "T1|T2|T4", 1, False,
        f"সরকারি ভাতা {s['amt']} টাকা পেতে আপনার {s['mfs']} পিন ও ওটিপি নম্বর পাঠান।")
    add("1_govt_subsidy_pin", "banglish", 1, "T1|T2|T4", 1, False,
        f"Shikkha britti {s['amt']} taka {s['mfs']} e nite apnar PIN ar OTP din.")
    add("1_govt_subsidy_pin", "mixed", 1, "T1|T2|T4", 1, False,
        f"Krishok/Student allowance Tk {s['amt']}: verify {s['mfs']} by sending PIN.")

    # 1_merchant_verify_pin
    add("1_merchant_verify_pin", "english", 1, "T1|T2", 1, False,
        f"{s['ecom']} Seller Center: Verify payout wallet by entering your {s['mfs']} PIN and OTP.")
    add("1_merchant_verify_pin", "banglish", 1, "T1|T2", 1, False,
        f"{s['ecom']} vendor payment {s['amt']} taka nite {s['mfs']} PIN verify korun.")
    add("1_merchant_verify_pin", "bangla", 1, "T1|T2", 1, False,
        f"{s['ecom']} পেমেন্ট রিসিভ করতে আপনার {s['mfs']} পিন ও ওটিপি দিন।")
    add("1_merchant_verify_pin", "mixed", 1, "T1|T2", 1, False,
        f"Payout hold on {s['ecom']}: share {s['mfs']} OTP to release Tk {s['amt']}.")

    # 1_fake_support_call
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Hello, {s['mfs']} head office theke bolchi. Apnar account e ekta complain asche.",
        "Ki complain?",
        "Keu apnar account hack korar chesta korche. Ami ekta code pathacchi, oita bolle block kore dibo."
    ], "greet")
    add("1_fake_support_call", "banglish", 1, "T1|T2|T5", 1, True, mt, n_t)
    add("1_fake_support_call", "english", 1, "T1|T2|T5", 5, True,
        f"{s['bank']} Fraud Dept: Call {s['phone']} immediately and provide your PIN to stop unauthorized login.")
    add("1_fake_support_call", "bangla", 1, "T1|T2|T5", 1, True,
        f"{s['mfs']} হেল্পডেস্ক: হ্যাকিং ঠেকাতে এখনই আপনার পিন ও ওটিপি বলুন।")
    add("1_fake_support_call", "mixed", 1, "T1|T2|T5", 1, True,
        f"{s['bank']} support: share the 6-digit security PIN sent to your phone.")

    # 1_biometric_sync_otp
    add("1_biometric_sync_otp", "english", 1, "T1|T2", 1, False,
        f"{s['mfs']} App Update: Sync biometric login by replying with your current PIN and OTP.")
    add("1_biometric_sync_otp", "bangla", 1, "T1|T2", 1, False,
        f"{s['mfs']} ফিঙ্গারপ্রিন্ট চালু করতে পিন ও ওটিপি রিপ্লাই করুন।")
    add("1_biometric_sync_otp", "banglish", 1, "T1|T2", 1, False,
        f"Face ID / Fingerprint update korte {s['mfs']} PIN ta pathan.")
    add("1_biometric_sync_otp", "mixed", 1, "T1|T2", 1, False,
        f"Biometric sync failed on {s['bank']}. Send OTP to complete.")

    # 1_security_breach_harvest
    add("1_security_breach_harvest", "english", 1, "T1|T2|T5", 1, True,
        f"ALERT: {s['mfs']} database breach! Secure your Tk {s['bal']} balance by sending your PIN and password now.")
    add("1_security_breach_harvest", "bangla", 1, "T1|T2|T5", 1, True,
        f"সতর্কতা: আপনার {s['mfs']} ব্যালেন্স সুরক্ষিত রাখতে এখনই পিন ও পাসওয়ার্ড পাঠান।")
    add("1_security_breach_harvest", "banglish", 1, "T1|T2|T5", 1, True,
        f"Server hack hoyeche! Taka bachate {s['mfs']} PIN ar OTP ekhoni pathan.")
    add("1_security_breach_harvest", "mixed", 1, "T1|T2|T5", 1, True,
        f"Protect your {s['bank']} funds: reply with card PIN immediately.")

    # =========================================================================
    # CLASS 2: 11 NEW SCENARIOS (+ 4 v1 = 15 scenarios total)
    # =========================================================================
    # 2_eid_gift_scam
    add("2_eid_gift_scam", "english", 2, "T1|T4|T5", 3, True,
        f"Eid Mubarak! {s['mfs']} is gifting Tk {s['amt']} Eidi to lucky users. Claim before midnight: {s['purl']}")
    add("2_eid_gift_scam", "bangla", 2, "T1|T4|T5", 3, True,
        f"ঈদ মোবারক! {s['telco']} থেকে {s['amt']} টাকা ঈদ উপহার জিতেছেন। নিতে ক্লিক করুন: {s['purl']}")
    add("2_eid_gift_scam", "banglish", 2, "T1|T4", 3, True,
        f"Eid gift! {s['ecom']} theke free gift box paben. Link e click kore claim korun: {s['purl']}")
    add("2_eid_gift_scam", "mixed", 2, "T4|T3", 2, True,
        f"Eid Salami Tk {s['amt']} won! Pay Tk 50 verification charge via {s['mfs']} to receive.")

    # 2_telco_bundle_gift
    add("2_telco_bundle_gift", "english", 2, "T1|T4|T5", 3, True,
        f"{s['telco']} Free 25GB Internet Gift! Claim your free data pack now at {s['purl']}.")
    add("2_telco_bundle_gift", "bangla", 2, "T1|T4|T5", 3, True,
        f"{s['telco']} ফ্রি ২০ জিবি ইন্টারনেট গিফট! অফারটি পেতে ১ ঘণ্টার মধ্যে ক্লিক করুন: {s['purl']}")
    add("2_telco_bundle_gift", "banglish", 2, "T1|T4", 3, True,
        f"Apnar {s['telco']} sim e free 15GB data gift asche! Claim korte {s['purl']} e jan.")
    add("2_telco_bundle_gift", "mixed", 2, "T1|T4|T5", 3, True,
        f"Congrats! {s['telco']} 500 min + 10GB free reward selected for you. Tap {s['purl']}.")

    # 2_survey_reward
    add("2_survey_reward", "english", 2, "T1|T4", 3, False,
        f"Answer 3 quick questions about {s['ecom']} and win a guaranteed Tk {s['amt']} gift card: {s['purl']}")
    add("2_survey_reward", "bangla", 2, "T1|T4", 3, False,
        f"{s['ecom']} সম্পর্কে ৩টি প্রশ্নের উত্তর দিয়ে জিতে নিন {s['amt']} টাকা পুরস্কার: {s['purl']}")
    add("2_survey_reward", "banglish", 2, "T4", 3, False,
        f"Choto ekta survey fill korun ar peye jan Tk {s['amt']} voucher: {s['purl']}")
    add("2_survey_reward", "mixed", 2, "T1|T4", 3, False,
        f"{s['mfs']} user feedback reward: answer 4 questions to win Tk {s['amt']} at {s['purl']}.")

    # 2_govt_grant_scam
    add("2_govt_grant_scam", "english", 2, "T1|T3|T4", 2, False,
        f"National Digital Grant: You are awarded Tk {s['amt']} cash support! Pay Tk 200 processing fee to claim.")
    add("2_govt_grant_scam", "bangla", 2, "T1|T4|T5", 3, False,
        f"সরকারি বিশেষ অনুদান {s['amt']} টাকা আপনার নামে এসেছে! পেতে এখনই আবেদন করুন: {s['purl']}")
    add("2_govt_grant_scam", "banglish", 2, "T1|T3|T4", 2, False,
        f"Govt grant Tk {s['amt']} pitechen! Fee Tk 150 {s['mfs']} kore claim korun {s['phone']} e.")
    add("2_govt_grant_scam", "mixed", 2, "T1|T4", 3, False,
        f"Youth Development Cash Award Tk {s['amt']} selected for your number! Claim: {s['purl']}")

    # 2_anniversary_reward
    add("2_anniversary_reward", "english", 2, "T1|T4|T5", 3, True,
        f"{s['ecom']} 10th Anniversary Celebration! You won a Tk {s['amt']} shopping voucher. Claim: {s['purl']}")
    add("2_anniversary_reward", "bangla", 2, "T1|T4", 3, True,
        f"{s['mfs']} বর্ষপূর্তি উপহার! আপনি জিতেছেন {s['amt']} টাকা বোনাস। লিংকে যান: {s['purl']}")
    add("2_anniversary_reward", "banglish", 2, "T1|T4", 3, True,
        f"{s['telco']} birthday gift: apni Tk {s['amt']} jitechen! Click {s['purl']}.")
    add("2_anniversary_reward", "mixed", 2, "T1|T4|T5", 3, True,
        f"Happy Anniversary from {s['bank']}! Claim your free Tk {s['amt']} reward today: {s['purl']}")

    # 2_scratch_card_win
    add("2_scratch_card_win", "english", 2, "T4|T5", 3, False,
        f"You received a digital scratch card worth up to Tk {s['amt']}! Scratch & claim within 2 hours: {s['purl']}")
    add("2_scratch_card_win", "bangla", 2, "T4", 3, False,
        f"আপনি একটি স্ক্র্যাচ কার্ড জিতেছেন ({s['amt']} টাকা পর্যন্ত)! সংগ্রহ করুন: {s['purl']}")
    add("2_scratch_card_win", "banglish", 2, "T4|T3", 2, False,
        f"Scratch card e laptop jitechen! Delivery charge Tk 350 pay kore claim korun.")
    add("2_scratch_card_win", "mixed", 2, "T4|T5", 4, False,
        f"Scratch & Win! Download reward app from {s['purl']}/win.apk to unlock Tk {s['amt']}.")

    # 2_referral_bonus_scam
    add("2_referral_bonus_scam", "english", 2, "T4|T5", 3, True,
        f"Congrats! Your number won the Tk {s['amt']} lucky referral bonus! Click {s['purl']} before expiry.")
    add("2_referral_bonus_scam", "bangla", 2, "T4", 3, True,
        f"অভিনন্দন! রেফারেল লটারিতে আপনি {s['amt']} টাকা জিতেছেন। লিংকে ক্লিক করে নিন: {s['purl']}")
    add("2_referral_bonus_scam", "banglish", 2, "T4|T3", 2, True,
        f"Bonus Tk {s['amt']} jitechen! Transfer fee Tk 100 pathan {s['phone']} e.")
    add("2_referral_bonus_scam", "mixed", 2, "T4", 5, True,
        f"Referral Jackpot Tk {s['amt']} won! WhatsApp {s['phone']} to receive prize.")

    # 2_quiz_win_prize
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Congratulations {s['name']}! Your number won our TV Quiz mega prize!",
        "Ami to kono quiz kheli nai.",
        f"Random mobile draw te select hoyechen. Tk {s['amt']} jitechen! Claim korte {s['purl']} e click korun."
    ], "greet")
    add("2_quiz_win_prize", "banglish", 2, "T4|T5", 3, False, mt, n_t)
    add("2_quiz_win_prize", "english", 2, "T3|T4", 2, False,
        f"Quiz Winner! You won an iPad Pro. Pay Tk {s['amt']} tax/registration fee to claim.")
    add("2_quiz_win_prize", "bangla", 2, "T4", 5, False,
        f"কুইজ বিজয়ী! আপনি স্মার্টটিভি জিতেছেন। নিতে {s['phone']} নম্বরে কল করুন।")
    add("2_quiz_win_prize", "mixed", 2, "T4", 1, False,
        f"TV Show Lucky Winner: reply with your full name, address & NID to claim Tk {s['amt']}.")

    # 2_festival_raffle
    add("2_festival_raffle", "english", 2, "T4|T3", 2, False,
        f"Pohela Boishakh Mega Raffle: You won Tk {s['amt']}! Pay Tk 200 stamp fee to release prize.")
    add("2_festival_raffle", "bangla", 2, "T4|T5", 3, False,
        f"বৈশাখী মেগা র‍্যাফেল ড্র-তে আপনি {s['amt']} টাকা জিতেছেন! এখনই ক্লেইম করুন: {s['purl']}")
    add("2_festival_raffle", "banglish", 2, "T4", 5, False,
        f"New Year raffle winner! Gold coin jitechen, WhatsApp korun {s['phone']}.")
    add("2_festival_raffle", "mixed", 2, "T1|T4", 3, False,
        f"{s['ecom']} Winter Raffle: Tk {s['amt']} gift card won! Claim at {s['purl']}.")

    # 2_vip_cash_prize
    add("2_vip_cash_prize", "english", 2, "T1|T4|T5", 3, False,
        f"VIP Member Reward: {s['mfs']} selected you for Tk {s['amt']} cash prize. Claim within 30 mins: {s['purl']}")
    add("2_vip_cash_prize", "bangla", 2, "T1|T4", 2, False,
        f"ভিআইপি পুরস্কার: {s['mfs']} থেকে {s['amt']} টাকা জিতেছেন। প্রসেসিং ফি ২০০ টাকা পাঠান।")
    add("2_vip_cash_prize", "banglish", 2, "T1|T4", 5, False,
        f"VIP customer draw te apni Tk {s['amt']} jitechen! Call {s['phone']} to claim.")
    add("2_vip_cash_prize", "mixed", 2, "T4", 1, False,
        f"VIP Prize Tk {s['amt']}: reply with your {s['mfs']} number and address to receive.")

    # 2_foreign_lottery
    add("2_foreign_lottery", "english", 2, "T3|T4|T5", 2, False,
        f"UK International Lottery: Your email/number won $10,000! Pay Tk {s['amt']} transfer clearance fee.")
    add("2_foreign_lottery", "bangla", 2, "T3|T4", 2, False,
        f"আন্তর্জাতিক লটারিতে আপনি ১০ লক্ষ টাকা জিতেছেন! ছাড়াতে {s['amt']} টাকা ফি দিন।")
    add("2_foreign_lottery", "banglish", 2, "T4", 5, False,
        f"Dubai lottery te apni Tk {s['amt']} jitechen! Claim korte {s['phone']} e WhatsApp korun.")
    add("2_foreign_lottery", "mixed", 2, "T4|T5", 3, False,
        f"Global Cash Prize won! Click {s['purl']} within 2h to claim your reward.")

    # =========================================================================
    # CLASS 3: 11 NEW SCENARIOS (+ 4 v1 = 15 scenarios total)
    # =========================================================================
    # 3_forex_bot
    add("3_forex_bot", "english", 3, "T5|T6", 4, False,
        f"Automated Forex AI Bot: Earn 40% daily profit passively! Join our Telegram channel: {s['purl']}")
    add("3_forex_bot", "bangla", 3, "T5|T6", 2, False,
        f"ফরেক্স ট্রেডিং বট: প্রতিদিন ৪০% নিশ্চিত লাভ! {s['amt']} টাকা ডিপোজিট করে শুরু করুন।")
    add("3_forex_bot", "banglish", 3, "T6", 4, False,
        f"Forex signal bot diye daily Tk {s['amt']} income korun. Join VIP: {s['purl']}")
    add("3_forex_bot", "mixed", 3, "T3|T6", 2, False,
        f"Auto Forex Trading: deposit Tk {s['amt']}, guaranteed weekly return 200%.")

    # 3_mlm_network
    add("3_mlm_network", "english", 3, "T3|T6", 2, False,
        f"Build your financial freedom! Join our network marketing team with Tk {s['amt']} starter kit and earn lakhs monthly.")
    add("3_mlm_network", "bangla", 3, "T3|T6", 4, False,
        f"নেটওয়ার্ক মার্কেটিংয়ে যোগ দিয়ে মাসে লক্ষ টাকা আয় করুন! জয়েন করতে লিংকে যান: {s['purl']}")
    add("3_mlm_network", "banglish", 3, "T6", 4, False,
        f"Matro 3 jon member join koralei Tk {s['amt']} bonus! Aj e group e join korun.")
    add("3_mlm_network", "mixed", 3, "T3|T6", 2, False,
        f"Diamond membership fee Tk {s['amt']}: recruit 2 friends and earn passive income.")

    # 3_loan_advance_fee
    add("3_loan_advance_fee", "english", 3, "T3|T6", 2, True,
        f"Instant Personal Loan up to Tk 5,00,000 without documents! Pay Tk {s['amt']} processing fee first via {s['mfs']}.")
    add("3_loan_advance_fee", "bangla", 3, "T3|T6", 2, True,
        f"বিনা জামানতে ৫ লক্ষ টাকা লোন নিন! প্রসেসিং ফি {s['amt']} টাকা আগে পাঠান।")
    add("3_loan_advance_fee", "banglish", 3, "T3|T6", 2, True,
        f"Easy loan Tk 2 lakh 1 ghontay! Registration fee Tk {s['amt']} {s['mfs']} korun {s['phone']} e.")
    add("3_loan_advance_fee", "mixed", 3, "T3|T6", 5, True,
        f"Zero-interest instant loan approved! Call {s['phone']} and pay Tk {s['amt']} stamp charge.")

    # 3_page_admin_job
    add("3_page_admin_job", "english", 3, "T6", 4, False,
        f"Hiring Facebook Page Admins! Work 1 hour/day from phone, earn Tk {s['amt']} daily. Apply on WhatsApp: {s['phone']}")
    add("3_page_admin_job", "bangla", 3, "T6", 4, False,
        f"ফেসবুক পেজ অ্যাডমিন নিয়োগ চলছে! দিনে ১ ঘণ্টা কাজে {s['amt']} টাকা আয়। হোয়াটসঅ্যাপে যোগ দিন।")
    add("3_page_admin_job", "banglish", 3, "T6", 4, False,
        f"Page admin lagbe, daily salary Tk {s['amt']}. Telegram group e join korun: {s['purl']}")
    add("3_page_admin_job", "mixed", 3, "T3|T6", 2, False,
        f"E-commerce page moderator job: pay Tk {s['amt']} ID activation fee to start.")

    # 3_brushing_review_task
    add("3_brushing_review_task", "english", 3, "T1|T6", 4, False,
        f"Earn Tk {s['amt']} per 5-star review on {s['ecom']} products! Join our Telegram task group: {s['purl']}")
    add("3_brushing_review_task", "bangla", 3, "T6", 4, False,
        f"প্রোডাক্টে ফাইভ স্টার রিভিউ দিয়ে প্রতিদিন {s['amt']} টাকা আয় করুন। টেলিগ্রামে যোগ দিন।")
    add("3_brushing_review_task", "banglish", 3, "T6", 4, False,
        f"Google Maps e review dilei protiti review te Tk {s['amt']}! WhatsApp korun {s['phone']}.")
    add("3_brushing_review_task", "mixed", 3, "T3|T6", 2, False,
        f"Prepaid merchant order task: deposit Tk {s['amt']}, get 30% commission back in 10 mins.")

    # 3_app_testing_job
    add("3_app_testing_job", "english", 3, "T6", 4, False,
        f"Get paid Tk {s['amt']} for each app you download and test! Install our task app: {s['purl']}/task.apk")
    add("3_app_testing_job", "bangla", 3, "T6", 4, False,
        f"অ্যাপ ডাউনলোড ও টেস্ট করে প্রতিদিন {s['amt']} টাকা আয় করুন। অ্যাপ নামান: {s['purl']}")
    add("3_app_testing_job", "banglish", 3, "T6", 4, False,
        f"Notun app install kore review dile Tk {s['amt']} income! Join {s['purl']}.")
    add("3_app_testing_job", "mixed", 3, "T6", 4, False,
        f"Mobile app tester needed: earn $15/hour. Join Telegram: {s['purl']}")

    # 3_captcha_typing_job
    add("3_captcha_typing_job", "english", 3, "T3|T6", 2, False,
        f"Simple Captcha Typing Job: Earn Tk {s['amt']} daily! Pay Tk 500 software/ID fee to start.")
    add("3_captcha_typing_job", "bangla", 3, "T3|T6", 2, False,
        f"ক্যাপচা টাইপিং কাজ: দিনে {s['amt']} টাকা আয়। আইডি খুলতে ৩০০ টাকা ফি দিন।")
    add("3_captcha_typing_job", "banglish", 3, "T6", 4, False,
        f"Mobile diye captcha typing kore Tk {s['amt']} income. WhatsApp group e join korun.")
    add("3_captcha_typing_job", "mixed", 3, "T3|T6", 2, False,
        f"Form filling home job: guaranteed Tk 25,000/month. Registration fee Tk {s['amt']}.")

    # 3_hotel_booking_task
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Hi, I am {s['name']} from {s['emp']} recruitment.",
        "Yes, what is the role?",
        f"We offer online hotel rating tasks. Earn Tk {s['amt']} per day in 20 minutes.",
        f"Join our Telegram mentor group to receive your first task: {s['purl']}"
    ], "casual")
    add("3_hotel_booking_task", "english", 3, "T1|T6", 4, False, mt, n_t)
    add("3_hotel_booking_task", "banglish", 3, "T6", 4, False,
        f"Hotel booking rating task kore daily Tk {s['amt']} income! Telegram e message din.")
    add("3_hotel_booking_task", "bangla", 3, "T3|T6", 2, False,
        f"অনলাইন হোটেল বুকিং টাস্ক: {s['amt']} টাকা রিচার্জ করে ৩০% কমিশন নিন।")
    add("3_hotel_booking_task", "mixed", 3, "T6", 4, False,
        f"Part-time resort rating job: earn Tk {s['amt']}/day. Join WhatsApp now.")

    # 3_telegram_vip_group
    add("3_telegram_vip_group", "english", 3, "T5|T6", 4, False,
        f"Secret Money-Making Telegram VIP Group! Limited 50 seats. Join now and earn Tk {s['amt']} today: {s['purl']}")
    add("3_telegram_vip_group", "bangla", 3, "T6", 4, False,
        f"টেলিগ্রাম ভিআইপি আর্নিং গ্রুপে যোগ দিন, প্রতিদিন {s['amt']} টাকা নিশ্চিত আয়: {s['purl']}")
    add("3_telegram_vip_group", "banglish", 3, "T6", 4, False,
        f"Daily earning Telegram channel e join korun, ajkei Tk {s['amt']} paben: {s['purl']}")
    add("3_telegram_vip_group", "mixed", 3, "T3|T6", 2, False,
        f"VIP Earning Club: pay Tk {s['amt']} entry fee for guaranteed daily payouts.")

    # 3_data_entry_job (hard negative twin of 0_interview_legit)
    add("3_data_entry_job", "english", 3, "T1|T3|T6", 2, True,
        f"HR, {s['emp']}: You are selected for remote Data Entry Officer! Pay Tk {s['amt']} training/laptop deposit to confirm.")
    add("3_data_entry_job", "bangla", 3, "T1|T3|T6", 2, True,
        f"{s['emp']} নিয়োগ: ঘরে বসে ডাটা এন্ট্রি চাকরির জন্য {s['amt']} টাকা জামানত পাঠান।")
    add("3_data_entry_job", "banglish", 3, "T1|T3|T6", 2, True,
        f"{s['emp']} e data entry job confirm korte Tk {s['amt']} registration fee {s['mfs']} korun.")
    add("3_data_entry_job", "mixed", 3, "T1|T3|T6", 5, True,
        f"Selected at {s['emp']}! Call {s['phone']} and pay Tk {s['amt']} ID card fee.")

    # 3_freelance_course_scam
    add("3_freelance_course_scam", "english", 3, "T3|T6", 2, True,
        f"100% Guaranteed $1,000/month Freelancing! Enroll in our secret course today for Tk {s['amt']}.")
    add("3_freelance_course_scam", "bangla", 3, "T3|T6", 2, True,
        f"১০০% নিশ্চিত মাসে ৫০,০০০ টাকা ফ্রিল্যান্সিং আয়! আজই {s['amt']} টাকা দিয়ে কোর্সে ভর্তি হোন।")
    add("3_freelance_course_scam", "banglish", 3, "T3|T6", 4, True,
        f"Dollar income course: matro {s['amt']} taka fee diye WhatsApp batch e join korun.")
    add("3_freelance_course_scam", "mixed", 3, "T3|T6", 2, True,
        f"Guaranteed Fiverr earning trick! Pay Tk {s['amt']} via {s['mfs']} to get access.")

    # =========================================================================
    # CLASS 4: 11 NEW SCENARIOS (+ 4 v1 = 15 scenarios total)
    # =========================================================================
    # 4_failed_delivery_redelivery
    add("4_failed_delivery_redelivery", "english", 4, "T1|T3|T5", 2, True,
        f"{s['courier']}: Delivery attempt #2 failed for {s['parcel']}. Pay Tk {s['amt']} redelivery fee at {s['purl']}")
    add("4_failed_delivery_redelivery", "bangla", 4, "T1|T3|T5", 2, True,
        f"{s['courier']}: আপনার পার্সেল {s['parcel']} ডেলিভারি করা যায়নি। পুনরায় পাঠাতে {s['amt']} টাকা ফি দিন: {s['purl']}")
    add("4_failed_delivery_redelivery", "banglish", 4, "T1|T3", 2, True,
        f"Delivery failed! Redelivery charge Tk {s['amt']} pay na korle parcel return hobe: {s['purl']}")
    add("4_failed_delivery_redelivery", "mixed", 4, "T1|T5", 3, True,
        f"{s['courier']} alert: reschedule delivery for #{s['parcel']} within 24h at {s['purl']}")

    # 4_cod_confirm_phish (hard negative twin of 0_cod_delivery_legit)
    add("4_cod_confirm_phish", "english", 4, "T1|T3", 2, True,
        f"{s['ecom']} Delivery: Your COD parcel #{s['parcel']} requires Tk {s['amt']} advance security deposit via {s['purl']} before dispatch.")
    add("4_cod_confirm_phish", "bangla", 4, "T1|T3", 2, True,
        f"আপনার সিওডি পার্সেল #{s['parcel']} পাঠানোর আগে {s['amt']} টাকা অগ্রিম পরিশোধ করুন: {s['purl']}")
    add("4_cod_confirm_phish", "banglish", 4, "T1|T3", 2, True,
        f"{s['courier']} COD parcel confirm korte age Tk {s['amt']} {s['mfs']} korun {s['phone']} e.")
    add("4_cod_confirm_phish", "mixed", 4, "T1|T3", 3, True,
        f"Confirm COD order #{s['parcel']} fee at {s['purl']} or order will be cancelled.")

    # 4_tracking_link_phish
    add("4_tracking_link_phish", "english", 4, "T1|T5", 3, True,
        f"{s['courier']}: Unusual tracking issue on parcel {s['parcel']}. Verify your delivery location immediately: {s['purl']}")
    add("4_tracking_link_phish", "bangla", 4, "T1|T5", 3, True,
        f"{s['courier']}: পার্সেল {s['parcel']} ট্র্যাকিং সমস্যা! লোকেশন নিশ্চিত করতে ক্লিক করুন: {s['purl']}")
    add("4_tracking_link_phish", "banglish", 4, "T1", 3, True,
        f"Apnar parcel {s['parcel']} track korte ar address confirm korte link e jan: {s['purl']}")
    add("4_tracking_link_phish", "mixed", 4, "T1|T5", 3, True,
        f"Live tracking paused for #{s['parcel']}. Update details at {s['purl']} today.")

    # 4_wrong_parcel_claim
    n_t, mt = _wrap_multiturn(rng, s, [
        f"Hello, this is {s['courier']} hub. A high-value parcel #{s['parcel']} arrived under your phone number.",
        "I didn't order anything.",
        f"Return cancellation requires verification. Click {s['purl']} or pay Tk {s['amt']} return fee."
    ], "casual")
    add("4_wrong_parcel_claim", "english", 4, "T1|T3|T5", 2, True, mt, n_t)
    add("4_wrong_parcel_claim", "banglish", 4, "T1|T3", 2, True,
        f"Apnar name e ekta international parcel asche. Cancel korte holeo Tk {s['amt']} charge dite hobe.")
    add("4_wrong_parcel_claim", "bangla", 4, "T1|T5", 3, True,
        f"ভুল ঠিকানার পার্সেল #{s['parcel']} বাতিল করতে এই লিংকে যান: {s['purl']}")
    add("4_wrong_parcel_claim", "mixed", 4, "T1|T3", 5, True,
        f"Unclaimed parcel #{s['parcel']} penalty! Call {s['phone']} to avoid fine.")

    # 4_parcel_vat_tax
    add("4_parcel_vat_tax", "english", 4, "T1|T3|T5", 2, False,
        f"Airport Parcel Clearance: Your package #{s['parcel']} owes Tk {s['amt']} VAT/Tax. Pay via {s['mfs']} {s['phone']} today.")
    add("4_parcel_vat_tax", "bangla", 4, "T1|T3|T5", 2, False,
        f"বিমানবন্দর কাস্টমস: আপনার পার্সেলে {s['amt']} টাকা ভ্যাট বকেয়া আছে। আজই পরিশোধ করুন।")
    add("4_parcel_vat_tax", "banglish", 4, "T1|T3", 2, False,
        f"Foreign parcel e Tk {s['amt']} tax baki ache. Clear korte {s['purl']} e pay korun.")
    add("4_parcel_vat_tax", "mixed", 4, "T1|T3|T5", 2, False,
        f"Customs VAT notice #{s['parcel']}: pay Tk {s['amt']} or parcel will be auctioned.")

    # 4_courier_kyc_verify
    add("4_courier_kyc_verify", "english", 4, "T1|T2|T5", 3, True,
        f"{s['courier']}: Courier KYC required to release parcel #{s['parcel']}. Submit NID and OTP at {s['purl']}")
    add("4_courier_kyc_verify", "bangla", 4, "T1|T2", 3, True,
        f"{s['courier']}: পার্সেল পেতে কেওয়াইসি যাচাই করুন। এনআইডি ও কোড দিন: {s['purl']}")
    add("4_courier_kyc_verify", "banglish", 4, "T1|T2", 1, True,
        f"Parcel {s['parcel']} release korte apnar NID number ar phone e asha code ta reply korun.")
    add("4_courier_kyc_verify", "mixed", 4, "T1|T5", 3, True,
        f"{s['courier']} identity check: verify parcel recipient at {s['purl']}.")

    # 4_intl_gift_customs
    add("4_intl_gift_customs", "english", 4, "T3|T4|T5", 2, False,
        f"A friend abroad sent you a luxury gift box held at Dhaka customs! Pay Tk {s['amt']} clearance charge to {s['phone']}.")
    add("4_intl_gift_customs", "bangla", 4, "T3|T4", 2, False,
        f"বিদেশ থেকে আপনার জন্য পাঠানো গিফট পার্সেল কাস্টমসে আটকে আছে। ছাড়াতে {s['amt']} টাকা পাঠান।")
    add("4_intl_gift_customs", "banglish", 4, "T3|T4", 2, False,
        f"London theke gift parcel asche, customs clearance Tk {s['amt']} {s['mfs']} korun.")
    add("4_intl_gift_customs", "mixed", 4, "T1|T3", 5, False,
        f"Diplomatic gift parcel #{s['parcel']} held at airport. Call agent {s['phone']}.")

    # 4_damaged_parcel_comp
    add("4_damaged_parcel_comp", "english", 4, "T1|T5", 3, False,
        f"{s['courier']}: Your parcel #{s['parcel']} was damaged in transit. Claim Tk {s['amt']} compensation at {s['purl']}")
    add("4_damaged_parcel_comp", "bangla", 4, "T1|T5", 3, False,
        f"{s['courier']}: আপনার পার্সেল ক্ষতিগ্রস্ত হয়েছে। ক্ষতিপূরণ পেতে ক্লিক করুন: {s['purl']}")
    add("4_damaged_parcel_comp", "banglish", 4, "T1", 3, False,
        f"Parcel damage compensation Tk {s['amt']} nite {s['purl']} e form fill korun.")
    add("4_damaged_parcel_comp", "mixed", 4, "T1|T3", 2, False,
        f"Claim parcel insurance Tk {s['amt']}: pay Tk 100 verification fee at {s['purl']}.")

    # 4_hub_pickup_fee
    add("4_hub_pickup_fee", "english", 4, "T1|T3", 2, True,
        f"{s['courier']}: Parcel {s['parcel']} is at local hub. Pay Tk {s['amt']} hub handling fee before pickup: {s['purl']}")
    add("4_hub_pickup_fee", "bangla", 4, "T1|T3", 2, True,
        f"{s['courier']}: হাব থেকে পার্সেল #{s['parcel']} তুলতে {s['amt']} টাকা সার্ভিস চার্জ দিন: {s['purl']}")
    add("4_hub_pickup_fee", "banglish", 4, "T1|T3", 2, True,
        f"Hub e parcel asche, Tk {s['amt']} gate pass fee {s['mfs']} kore collect korun.")
    add("4_hub_pickup_fee", "mixed", 4, "T1|T3", 3, True,
        f"Pay Tk {s['amt']} locker fee to unlock parcel #{s['parcel']} at {s['purl']}.")

    # 4_warehouse_penalty
    add("4_warehouse_penalty", "english", 4, "T1|T3|T5", 2, True,
        f"FINAL NOTICE: Parcel #{s['parcel']} incurring Tk {s['amt']}/day warehouse penalty! Pay now at {s['purl']}")
    add("4_warehouse_penalty", "bangla", 4, "T1|T3|T5", 2, True,
        f"শেষ সতর্কতা: পার্সেল #{s['parcel']} গুদামে রাখায় প্রতিদিন {s['amt']} টাকা জরিমানা হচ্ছে। এখনই পরিশোধ করুন।")
    add("4_warehouse_penalty", "banglish", 4, "T1|T3|T5", 2, True,
        f"Warehouse charge Tk {s['amt']} hoye gese! Ajke pay na korle legal action: {s['purl']}")
    add("4_warehouse_penalty", "mixed", 4, "T1|T5", 5, True,
        f"Overdue parcel storage #{s['parcel']}: call {s['phone']} immediately.")

    # 4_courier_pin_confirm
    add("4_courier_pin_confirm", "english", 4, "T1|T2", 1, False,
        f"{s['courier']}: To confirm delivery of parcel #{s['parcel']}, reply with the 6-digit OTP sent to your phone.")
    add("4_courier_pin_confirm", "bangla", 4, "T1|T2", 1, False,
        f"{s['courier']}: পার্সেল #{s['parcel']} গ্রহণ নিশ্চিত করতে ফোনে আসা ওটিপি কোডটি বলুন।")
    add("4_courier_pin_confirm", "banglish", 4, "T1|T2", 1, False,
        f"Parcel release korte apnar phone e asha verification code ta reply korun.")
    add("4_courier_pin_confirm", "mixed", 4, "T1|T2", 3, False,
        f"Confirm delivery #{s['parcel']} by entering OTP at {s['purl']}.")

    # =========================================================================
    # CLASS 5: 16 NEW SCENARIOS (+ 4 v1 = 20 scenarios total)
    # =========================================================================
    # 5_restaurant_promo
    add("5_restaurant_promo", "english", 5, "NONE", 0, False,
        f"Pizza Hut Weekend Combo: 2 large pizzas + garlic bread for Tk {s['amt']} today only! Order in app.")
    add("5_restaurant_promo", "bangla", 5, "NONE", 0, False,
        f"বিরিয়ানি উৎসব! আজ অর্ডার করলেই পাচ্ছেন ২০% ডিসকাউন্ট, কোড: FOOD20।")
    add("5_restaurant_promo", "banglish", 5, "NONE", 0, False,
        f"Burger Lover! Buy 1 get 1 free on all combos till 10 PM at Dhanmondi outlet.")
    add("5_restaurant_promo", "mixed", 5, "NONE", 0, False,
        f"KFC Wednesday deal: 8 pcs hot & crispy Tk {s['amt']}. Visit nearest branch.")

    # 5_telco_bundles
    add("5_telco_bundles", "english", 5, "NONE", 0, True,
        f"{s['telco']}: Special 10GB + 200 min bundle for Tk {s['amt']}, valid 7 days! Dial *121*5# to buy.")
    add("5_telco_bundles", "bangla", 5, "NONE", 0, True,
        f"{s['telco']} দারুণ অফার! {s['amt']} টাকায় ১৫ জিবি ইন্টারনেট ৭ দিন মেয়াদে। ডায়াল *১২১#।")
    add("5_telco_bundles", "banglish", 5, "NONE", 0, True,
        f"{s['telco']} combo pack: 5GB + 100 min matro {s['amt']} taka! MyApp theke kinun.")
    add("5_telco_bundles", "mixed", 5, "NONE", 0, True,
        f"{s['telco']} night pack 20GB at Tk {s['amt']}. Dial *5000# to activate.")

    # 5_flash_sale
    add("5_flash_sale", "english", 5, "NONE", 0, True,
        f"{s['ecom']} 11.11 Flash Sale starts at midnight! Discounts up to 60% + free shipping. Shop: {s['lurl']}")
    add("5_flash_sale", "bangla", 5, "NONE", 0, True,
        f"{s['ecom']} ফ্ল্যাশ সেল! আজ রাত ১২টা থেকে সব পণ্যে ৫০% পর্যন্ত ছাড়: {s['lurl']}")
    add("5_flash_sale", "banglish", 5, "NONE", 0, True,
        f"Mega Flash Sale! Ajke rat 12 tay {s['ecom']} e 40% off. Voucher code FLASH40.")
    add("5_flash_sale", "mixed", 5, "NONE", 0, True,
        f"Limited time flash deal at {s['ecom']}! Save Tk {s['amt']} on electronics.")

    # 5_insurance_offer
    add("5_insurance_offer", "english", 5, "NONE", 0, False,
        f"{s['bank']} Life Care: Secure your family's future with health insurance starting Tk {s['amt']}/month. Call {s['phone']}.")
    add("5_insurance_offer", "bangla", 5, "NONE", 0, False,
        f"স্বাস্থ্য সুরক্ষা বীমা: মাসে মাত্র {s['amt']} টাকায় পরিবারের চিকিৎসা কভারেজ নিন।")
    add("5_insurance_offer", "banglish", 5, "NONE", 0, False,
        f"Family health insurance plan, bochore 5 lakh cover. Details jante {s['lurl']} dekhun.")
    add("5_insurance_offer", "mixed", 5, "NONE", 0, False,
        f"Protect your car with {s['bank']} auto insurance. 15% early renewal discount.")

    # 5_bank_loan_offer (hard negative twin of 3_loan_advance_fee)
    add("5_bank_loan_offer", "english", 5, "NONE", 0, True,
        f"{s['bank']}: Pre-approved personal loan at 9.5% interest! Apply at your nearest branch or {s['lurl']}. T&C apply.")
    add("5_bank_loan_offer", "bangla", 5, "NONE", 0, True,
        f"{s['bank']} হোম লোন অফার: সহজ কিস্তিতে বাড়ি বা ফ্ল্যাট কিনুন। বিস্তারিত জানতে শাখায় যোগাযোগ করুন।")
    add("5_bank_loan_offer", "banglish", 5, "NONE", 0, True,
        f"{s['bank']} credit card ar auto loan offer! 0% processing fee this month. Apply in app.")
    add("5_bank_loan_offer", "mixed", 5, "NONE", 0, True,
        f"SME business loan from {s['bank']}. Visit {s['lurl']} for eligibility criteria.")

    # 5_event_invitation
    add("5_event_invitation", "english", 5, "NONE", 0, False,
        f"You're invited to Dhaka Tech Expo 2026 at ICCB! Register your free visitor pass at {s['lurl']}.")
    add("5_event_invitation", "bangla", 5, "NONE", 0, False,
        f"ঢাকা বইমেলা ও সাহিত্য উৎসবে আপনাকে আমন্ত্রণ! স্টল নম্বর ৪৫ ভিজিট করুন।")
    add("5_event_invitation", "banglish", 5, "NONE", 0, False,
        f"Free Career Seminar this Saturday at Banani. Seat book korte {s['lurl']} e jan.")
    add("5_event_invitation", "mixed", 5, "NONE", 0, False,
        f"Live Music Fest tickets now available! Early bird 20% discount with code MUSIC20.")

    # 5_gym_fitness
    add("5_gym_fitness", "english", 5, "NONE", 0, False,
        f"FitZone Gym Gulshan: 3-month membership at Tk {s['amt']}! Free trainer assessment this week.")
    add("5_gym_fitness", "bangla", 5, "NONE", 0, False,
        f"ফিটনেস ক্লাবে ভর্তি চলছে! প্রথম মাসের ফি-তে ৩০% ছাড়।")
    add("5_gym_fitness", "banglish", 5, "NONE", 0, False,
        f"Summer fitness offer: 6 month gym package matro Tk {s['amt']}. Call {s['phone']}.")
    add("5_gym_fitness", "mixed", 5, "NONE", 0, False,
        f"Yoga & Zumba morning batch open! Free trial class Friday.")

    # 5_travel_deal
    add("5_travel_deal", "english", 5, "NONE", 0, False,
        f"Cox's Bazar Weekend Package: 2 nights hotel + flight starting Tk {s['amt']}! Book at {s['lurl']}.")
    add("5_travel_deal", "bangla", 5, "NONE", 0, False,
        f"সাজেক ও কক্সবাজার ভ্রমণ প্যাকেজ! গ্রুপ বুকিংয়ে ১৫% ছাড়। কল করুন {s['phone']}।")
    add("5_travel_deal", "banglish", 5, "NONE", 0, False,
        f"Sylhet resort booking e 25% discount! Use promo SYLHET25.")
    add("5_travel_deal", "mixed", 5, "NONE", 0, False,
        f"Domestic flight tickets starting Tk {s['amt']}. Book early via official app.")

    # 5_electronics_sale
    add("5_electronics_sale", "english", 5, "NONE", 0, True,
        f"Mega Electronics Clearance: Up to 35% off smartphones & laptops at {s['ecom']} showroom! T&C apply.")
    add("5_electronics_sale", "bangla", 5, "NONE", 0, True,
        f"শীতকালীন ইলেকট্রনিক্স মেলা! এসি ও ফ্রিজে ২০% ডিসকাউন্ট ও ফ্রি হোম ডেলিভারি।")
    add("5_electronics_sale", "banglish", 5, "NONE", 0, True,
        f"Smart TV kinle soundbar free! Offer cholbe ei soptaho porjonto.")
    add("5_electronics_sale", "mixed", 5, "NONE", 0, True,
        f"0% EMI on laptops for 12 months with {s['bank']} cards at {s['ecom']}.")

    # 5_fashion_collection
    add("5_fashion_collection", "english", 5, "NONE", 0, False,
        f"{s['ecom']} New Winter Collection is here! Flat 25% off jackets and hoodies in-store and online.")
    add("5_fashion_collection", "bangla", 5, "NONE", 0, False,
        f"নতুন ঈদ কালেকশন এখন সব শোরুমে! পাঞ্জাবি ও শাড়িতে আকর্ষণীয় মূল্যছাড়।")
    add("5_fashion_collection", "banglish", 5, "NONE", 0, False,
        f"End of season fashion sale! Sob panjabi ar kurti te 40% off.")
    add("5_fashion_collection", "mixed", 5, "NONE", 0, False,
        f"Buy 2 get 1 free on formal shirts at {s['ecom']} outlet.")

    # 5_grocery_promo
    add("5_grocery_promo", "english", 5, "NONE", 0, False,
        f"{s['ecom']} Grocery: Free delivery on orders above Tk 999! Fresh vegetables & fruits delivered in 1 hour.")
    add("5_grocery_promo", "bangla", 5, "NONE", 0, False,
        f"তাজা বাজার ও গ্রোসারি অর্ডারে ফ্রি ডেলিভারি! কোড ব্যবহার করুন: FRESH।")
    add("5_grocery_promo", "banglish", 5, "NONE", 0, False,
        f"Monthly bazar e Tk {s['amt']} instant discount! Order korun {s['ecom']} app theke.")
    add("5_grocery_promo", "mixed", 5, "NONE", 0, False,
        f"Weekend grocery deal: 10% off cooking oil and rice.")

    # 5_rideshare_promo
    add("5_rideshare_promo", "english", 5, "NONE", 0, False,
        f"Ride & Save! Get 30% off your next 3 bike or car rides with code RIDE30.")
    add("5_rideshare_promo", "bangla", 5, "NONE", 0, False,
        f"অফিস যাওয়ার রাইডে ৪০% ছাড়! আজই অ্যাপ থেকে রাইড বুক করুন।")
    add("5_rideshare_promo", "banglish", 5, "NONE", 0, False,
        f"Ajke sob bike ride e flat Tk 50 off! Promo code DHAKA50.")
    add("5_rideshare_promo", "mixed", 5, "NONE", 0, False,
        f"Zero surge pricing during afternoon hours! Book your ride now.")

    # 5_ott_streaming_deal
    add("5_ott_streaming_deal", "english", 5, "NONE", 0, False,
        f"Watch new Bangla web series & live cricket! Subscribe for Tk {s['amt']}/month at {s['lurl']}.")
    add("5_ott_streaming_deal", "bangla", 5, "NONE", 0, False,
        f"নতুন বাংলা সিনেমা ও লাইভ খেলা দেখুন! মাসিক প্যাক মাত্র {s['amt']} টাকা।")
    add("5_ott_streaming_deal", "banglish", 5, "NONE", 0, False,
        f"Annual OTT subscription e 50% off! Ajkei subscribe korun.")
    add("5_ott_streaming_deal", "mixed", 5, "NONE", 0, False,
        f"7-day free trial for premium movies and sports channels.")

    # 5_card_reward_promo
    add("5_card_reward_promo", "english", 5, "NONE", 0, True,
        f"{s['bank']} Cardholders: Earn 2x reward points on all dining & shopping spends this weekend. T&C apply.")
    add("5_card_reward_promo", "bangla", 5, "NONE", 0, True,
        f"{s['bank']} কার্ডে শপিং করলে ১০% ইনস্ট্যান্ট ক্যাশব্যাক! শর্ত প্রযোজ্য।")
    add("5_card_reward_promo", "banglish", 5, "NONE", 0, True,
        f"{s['mfs']} payment e 15% cashback paben {s['ecom']} e, max Tk 200.")
    add("5_card_reward_promo", "mixed", 5, "NONE", 0, True,
        f"Airport lounge access + 10% cashback with {s['bank']} Platinum Card.")

    # 5_health_checkup_pkg
    add("5_health_checkup_pkg", "english", 5, "NONE", 0, False,
        f"Full Body Health Checkup Package for Tk {s['amt']} (30% off) this month! Book appointment: {s['phone']}")
    add("5_health_checkup_pkg", "bangla", 5, "NONE", 0, False,
        f"ডায়াবেটিস ও হার্ট চেকআপ প্যাকেজে ২৫% ছাড়! আজই বুকিং দিন।")
    add("5_health_checkup_pkg", "banglish", 5, "NONE", 0, False,
        f"Senior citizen health checkup package Tk {s['amt']}. Call {s['phone']}.")
    add("5_health_checkup_pkg", "mixed", 5, "NONE", 0, False,
        f"Dental scaling & polishing 40% discount offer till Friday.")

    # 5_furniture_sale
    add("5_furniture_sale", "english", 5, "NONE", 0, False,
        f"Upgrade your home! Flat 20% discount on sofas, beds, and dining sets at our showroom.")
    add("5_furniture_sale", "bangla", 5, "NONE", 0, False,
        f"ঘর সাজান নতুন ফার্নিচারে! সব খাট ও সোফায় ২০% ছাড় ও ফ্রি ফিটিং।")
    add("5_furniture_sale", "banglish", 5, "NONE", 0, False,
        f"Home decor & furniture exhibition e 30% discount cholche!")
    add("5_furniture_sale", "mixed", 5, "NONE", 0, False,
        f"Office chair & desk combo offer Tk {s['amt']}. Visit Mirpur showroom.")

    return rows


def paraphrase_real_row(row: pd.Series, variant_idx: int, rng: random.Random) -> dict:
    """Create genuinely rephrased/restructured paraphrase variants of a real sample.csv row."""
    txt = str(row["text"]).strip()
    lbl = int(row["label"])
    tech = str(row["techniques"])
    act = int(row["action"])
    rid = int(row["id"])

    # Detect language roughly
    has_bn = bool(re.search(r"[\u0980-\u09FF]", txt))
    lang = "bangla" if has_bn else rng.choice(["english", "banglish", "mixed"])

    # Clean turns if multi-turn
    turns = [t.strip() for t in txt.split("|||")]
    is_mt = len(turns) > 1

    # Restructuring strategies based on variant_idx (0..9)
    if variant_idx % 5 == 0:
        # Reverse/front-load call-to-action or add contextual framing
        if is_mt:
            spk = rng.choice(["Agent", "Caller", "Sender"])
            other = rng.choice(["Customer", "You", "Receiver"])
            new_turns = [f"{spk}: Hello, ekta kotha chilo.", f"{other}: Ji bolun."]
            for i, t in enumerate(turns):
                clean_t = re.sub(r"^(S|R|A|B|Caller|You|Agent|Customer):\s*", "", t)
                new_turns.append(f"{spk if i % 2 == 0 else other}: {clean_t}")
            new_txt = " ||| ".join(new_turns)
            n_t = len(new_turns)
        else:
            prefix = rng.choice([
                "Notice: ", "Attention: ", "Update: ", "Info: ", "জরুরি বার্তা: ", "Shunun: "
            ]) if not has_bn else rng.choice(["জরুরি বার্তা: ", "তথ্য: ", "বিজ্ঞপ্তি: "])
            suffix = rng.choice([
                " Please check.", " Kind regards.", " Thanks.", " ধন্যবাদ।" if has_bn else " Dhonyobad."
            ])
            new_txt = prefix + txt + suffix
            n_t = 1
    elif variant_idx % 5 == 1:
        # Convert single-turn to 3-turn conversation (or expand existing multi-turn)
        if not is_mt:
            spk_s, spk_r = rng.choice([("S", "R"), ("Caller", "You"), ("A", "B")])
            new_txt = f"{spk_s}: Salam, free achen? ||| {spk_r}: Ji bolun. ||| {spk_s}: {txt}"
            n_t = 3
        else:
            # Drop speaker tags and rephrase into compact narrative
            joined = " ".join(re.sub(r"^(S|R|A|B|Caller|You|Agent|Customer):\s*", "", t) for t in turns)
            new_txt = f"Conversation summary: {joined}"
            n_t = 1
    elif variant_idx % 5 == 2:
        # Synonym substitution + sentence-level rephrasing
        swaps = [
            ("successful", "completed"), ("Successful", "Completed"),
            ("Do not share", "Never disclose"), ("never ask for", "do not request"),
            ("Click to claim", "Tap the link to collect"), ("click", "visit"),
            ("Pay", "Send"), ("pay", "transfer"),
            ("discount", "price off"), ("offer", "deal"), ("Offer", "Deal"),
            ("held", "stopped"), ("expires", "ends"), ("within 1 hour", "before 60 minutes"),
            ("bolen", "janan"), ("pathan", "send korun"), ("korun", "koren"),
            ("taka", "BDT"), ("ferot din", "return korun"), ("jitechen", "won"),
            ("সফল হয়েছে", "সম্পন্ন হয়েছে"), ("ক্লিক করুন", "ভিজিট করুন"),
            ("পাঠান", "শেয়ার করুন"), ("টাকা", "টাকা"),
        ]
        new_txt = txt
        for a, b in swaps:
            if a in new_txt and rng.random() < 0.7:
                new_txt = new_txt.replace(a, b, 1)
        if new_txt == txt:
            new_txt = f"Message received: {txt}"
        n_t = len(turns)
    elif variant_idx % 5 == 3:
        # Clause reordering if sentence has two parts
        parts = re.split(r"(?<=[.!।])\s+", txt)
        if len(parts) >= 2 and not is_mt:
            reordered = parts[-1] + " " + " ".join(parts[:-1])
            new_txt = reordered.strip()
        else:
            new_txt = f"FYI - {txt}"
        n_t = len(turns)
    else:
        # Conversational inquiry wrapper
        if has_bn:
            new_txt = f"S: ভাই একটা মেসেজ দেখলাম। ||| R: কী মেসেজ? ||| S: {txt}"
            n_t = 3
        else:
            new_txt = f"A: Hey, got this message. ||| B: What does it say? ||| A: {txt}"
            n_t = 3

    return {
        "text": new_txt,
        "label": lbl,
        "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
        "action": act,
        "family_id": f"real_para_{rid}",
        "scenario_id": f"real_para_{rid}",
        "language": lang,
        "n_turns": n_t,
        "is_hard_negative": lbl in (0, 1, 4, 5),
        "source": "real_paraphrase",
    }


def generate_synthetic_v2(target_rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    # 1. Generate 10 paraphrases per real row (860 rows)
    real_df = pd.read_csv("sample.csv")
    paraphrase_rows = []
    for _, row in real_df.iterrows():
        for v_idx in range(10):
            paraphrase_rows.append(paraphrase_real_row(row, v_idx, rng))

    # 2. Remaining target (~14,140) filled from v1 + expanded v2 scenarios
    rem_target = target_rows - len(paraphrase_rows)
    # Balanced across all 6 classes (with slight extra weight for 0 and 5)
    target_per_class = {
        0: int(rem_target * 0.18),
        1: int(rem_target * 0.16),
        2: int(rem_target * 0.16),
        3: int(rem_target * 0.16),
        4: int(rem_target * 0.16),
        5: rem_target - int(rem_target * 0.18) - 4 * int(rem_target * 0.16),
    }

    buckets = {k: [] for k in range(6)}

    while any(len(buckets[k]) < target_per_class[k] for k in range(6)):
        slots = _build_slots(rng)

        # Collect v1 templates
        for fam, lang, lbl, tech, act, hard, txt in _gen_single_templates(slots):
            if len(buckets[lbl]) < target_per_class[lbl] and rng.random() < 0.25:
                buckets[lbl].append({
                    "text": txt,
                    "label": lbl,
                    "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
                    "action": act,
                    "family_id": fam,
                    "scenario_id": V1_FAMILY_TO_SCENARIO[fam],
                    "language": lang,
                    "n_turns": 1,
                    "is_hard_negative": hard,
                    "source": "synthetic_v1",
                })

        for fam, lang, lbl, tech, act, hard, n_t, txt in _gen_multiturn_templates(rng, slots):
            if len(buckets[lbl]) < target_per_class[lbl] and rng.random() < 0.45:
                buckets[lbl].append({
                    "text": txt,
                    "label": lbl,
                    "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
                    "action": act,
                    "family_id": fam,
                    "scenario_id": V1_FAMILY_TO_SCENARIO[fam],
                    "language": lang,
                    "n_turns": n_t,
                    "is_hard_negative": hard,
                    "source": "synthetic_v1",
                })

        # Collect new v2 scenarios (both single and multi-turn wrapped variants to hit ~35% multi-turn)
        for scen, fam, lang, lbl, tech, act, n_t, hard, txt in _gen_new_scenarios(rng, slots):
            if len(buckets[lbl]) >= target_per_class[lbl]:
                continue
            # Also randomly wrap ~28% of single-turn new scenarios into multi-turn delayed reveal
            if n_t == 1 and rng.random() < 0.28:
                n_t, txt = _wrap_multiturn(rng, slots, [txt], rng.choice(["greet", "casual", "bangla"]))
            buckets[lbl].append({
                "text": txt,
                "label": lbl,
                "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
                "action": act,
                "family_id": fam,
                "scenario_id": scen,
                "language": lang,
                "n_turns": n_t,
                "is_hard_negative": hard,
                "source": "synthetic_v2",
            })

    all_rows = paraphrase_rows[:]
    for k in range(6):
        all_rows.extend(buckets[k][:target_per_class[k]])

    rng.shuffle(all_rows)
    df = pd.DataFrame(all_rows)

    # Apply noise to ~30% of rows
    is_noisy = []
    noisy_texts = []
    for txt in df["text"].tolist():
        if rng.random() < 0.30:
            noisy_texts.append(apply_noise(txt, rng, level=0.35))
            is_noisy.append(True)
        else:
            noisy_texts.append(txt)
            is_noisy.append(False)
    df["text"] = noisy_texts
    df["is_noisy"] = is_noisy

    df.insert(0, "id", range(200001, 200001 + len(df)))
    return df


if __name__ == "__main__":
    df = generate_synthetic_v2(target_rows=15000, seed=42)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_v2.csv", index=False)
    print("Saved data/synthetic_v2.csv:", df.shape)
    print("Labels:\n", df["label"].value_counts().sort_index().to_dict())
    print("Languages:\n", df["language"].value_counts().to_dict())
    print("Multi-turn ratio:", round((df["n_turns"] > 1).mean(), 4))
    print("Noisy ratio:", round(df["is_noisy"].mean(), 4))
    print("Sources:\n", df["source"].value_counts().to_dict())
    for lbl in range(6):
        scens = df[(df["label"] == lbl) & (df["source"] != "real_paraphrase")]["scenario_id"].nunique()
        print(f"  Class {lbl} distinct scenarios: {scens}")
