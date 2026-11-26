import random
import pandas as pd

# Real + Invented Bangladeshi and generic brands/senders
MFS_BRANDS = [
    "bKash", "Nagad", "Rocket", "Upay", "SureCash", "Tap", "Cellfin",
    "PayBondhu", "TakaSwift", "DeshPay", "ChakaPay", "SonarWallet"
]
BANK_BRANDS = [
    "DBBL", "City Bank", "BRAC Bank", "Islami Bank", "Standard Chartered",
    "UCB", "EBL", "Mutual Trust Bank", "NodiBank", "ApexFin BD", "PadmaTrust Bank"
]
COURIER_BRANDS = [
    "Sundarban Courier", "SA Paribahan", "Pathao Courier", "RedX", "Paperfly",
    "Steadfast", "Karatoa", "QuickParcel BD", "SonarCourier", "DhakaExpress"
]
TELCO_BRANDS = [
    "Grameenphone", "GP", "Robi", "Banglalink", "Teletalk", "PadmaTel", "BongoNet"
]
ECOM_BRANDS = [
    "Daraz", "Shwapno", "Aarong", "Chaldal", "Pickaboo", "Rokomari",
    "MeghnaMart", "ShebaHaat", "BongoBazar"
]
EMPLOYER_BRANDS = [
    "Unilever Bangladesh", "Bdjobs", "GrameenTech", "Square Group", "Walton",
    "ApexDigital BD", "EarningHub BD", "DeshWorks"
]
CRYPTO_BRANDS = [
    "Binance", "CoinBangla", "BDTradeX", "AlphaCrypto", "DhakaSignals"
]
PERSON_NAMES = [
    "Rahim", "Karim", "Sadia", "Nusrat", "Tanvir", "Tariq", "Farhan", "Mitu", "Hasan", "Nabila"
]
PHISH_DOMAINS = [
    "verify-bd[.]top", "claim-now[.]xyz", "secure-kyc[.]site", "pay-fee[.]vip",
    "track-parcel[.]top", "bonus-bd[.]info", "vip-earn[.]xyz", "confirm-addr[.]top"
]
LEGIT_DOMAINS = [
    "daraz[.]com[.]bd/track", "payment[.]breb[.]org[.]bd", "mygp[.]grameenphone[.]com",
    "bkash[.]com/help", "sundarbancourier[.]com[.]bd", "bracbank[.]com/portal"
]


def _rand_amt(rng: random.Random) -> str:
    return rng.choice(["45", "50", "100", "250", "300", "500", "750", "1,000", "1,250", "1,500", "2,000", "2,500", "3,500", "5,000", "10,000", "50,000"])


def _rand_phone(rng: random.Random) -> str:
    prefix = rng.choice(["017", "018", "019", "016", "013", "015"])
    return f"{prefix}XXXXXX{rng.randint(10, 99)}"


def _rand_trx(rng: random.Random) -> str:
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(chars) for _ in range(8))


def _rand_otp(rng: random.Random) -> str:
    return str(rng.randint(100000, 999999))


def _rand_parcel(rng: random.Random) -> str:
    return f"{rng.choice(['SP', 'SC', 'DZ', 'PF', 'CN', 'RX'])}{rng.randint(1000, 99999)}"


def _build_slots(rng: random.Random) -> dict:
    return {
        "mfs": rng.choice(MFS_BRANDS),
        "bank": rng.choice(BANK_BRANDS),
        "courier": rng.choice(COURIER_BRANDS),
        "telco": rng.choice(TELCO_BRANDS),
        "ecom": rng.choice(ECOM_BRANDS),
        "emp": rng.choice(EMPLOYER_BRANDS),
        "crypto": rng.choice(CRYPTO_BRANDS),
        "name": rng.choice(PERSON_NAMES),
        "amt": _rand_amt(rng),
        "bal": str(rng.randint(500, 25000)),
        "phone": _rand_phone(rng),
        "trx": _rand_trx(rng),
        "otp": _rand_otp(rng),
        "parcel": _rand_parcel(rng),
        "purl": f"hxxps://{rng.choice(PHISH_DOMAINS)}/{rng.choice(['verify', 'claim', 'pay', 'update'])}",
        "lurl": f"https://{rng.choice(LEGIT_DOMAINS)}",
    }


def _gen_single_templates(s: dict) -> list:
    """Return list of (family_id, language, label, techniques, action, is_hard_negative, text)."""
    rows = []

    # ==================== CLASS 0: LEGITIMATE (HARD LEGIT + PERSONAL) ====================
    # 1. Legit OTP with do-not-share warning (4 languages)
    rows.append(("c0_otp_warn", "english", 0, "NONE", 0, True,
                 f"{s['mfs']} OTP is {s['otp']}. Do not share this code with anyone, including {s['mfs']} staff. Valid 3 min."))
    rows.append(("c0_otp_warn", "bangla", 0, "NONE", 0, True,
                 f"আপনার {s['mfs']} ওটিপি {s['otp']}। এই কোড কারো সাথে শেয়ার করবেন না, এমনকি {s['mfs']} কর্মীকেও না।"))
    rows.append(("c0_otp_warn", "banglish", 0, "NONE", 0, True,
                 f"{s['bank']} OTP: {s['otp']}. Eta kauke bolben na. Bank er lokeo PIN ba OTP chay na. Meyad 3 minute."))
    rows.append(("c0_otp_warn", "mixed", 0, "NONE", 0, True,
                 f"{s['mfs']} login code {s['otp']} for Tk {s['amt']} payment. Kauke share korben na."))

    # 2. Legit Cash-out / Receive / Debit confirmation
    rows.append(("c0_tx_confirm", "english", 0, "NONE", 0, True,
                 f"{s['mfs']}: You have received Tk {s['amt']} from {s['phone']}. Fee Tk 0.00. Balance Tk {s['bal']}. TrxID {s['trx']}."))
    rows.append(("c0_tx_confirm", "bangla", 0, "NONE", 0, True,
                 f"আপনার {s['amt']} টাকা ক্যাশ আউট সফল হয়েছে। বর্তমান ব্যালেন্স {s['bal']} টাকা। TrxID {s['trx']}।"))
    rows.append(("c0_tx_confirm", "banglish", 0, "NONE", 0, True,
                 f"Dhonyobad! Apnar Tk {s['amt']} recharge successful. Balance Tk {s['bal']}. TrxID {s['trx']}."))
    rows.append(("c0_tx_confirm", "mixed", 0, "NONE", 0, True,
                 f"{s['bank']}: BDT {s['amt']} debited from A/C XXXX{s['otp'][:4]} at {s['ecom']}. Avl bal BDT {s['bal']}."))

    # 3. Legit Delivery / Parcel updates (no fee demanded)
    rows.append(("c0_parcel_legit", "english", 0, "NONE", 0, True,
                 f"Your {s['ecom']} order #{s['parcel']} is out for delivery today. Track: {s['lurl']}"))
    rows.append(("c0_parcel_legit", "bangla", 0, "NONE", 0, True,
                 f"{s['courier']}: আপনার পার্সেল {s['parcel']} আজ ডেলিভারি হবে। শাখা থেকে সংগ্রহ করতে এই এসএমএস ও এনআইডি আনুন।"))
    rows.append(("c0_parcel_legit", "banglish", 0, "NONE", 0, True,
                 f"{s['courier']}: Parcel {s['parcel']} Chattogram hub e pouchheche. Branch theke collect korun."))
    rows.append(("c0_parcel_legit", "mixed", 0, "NONE", 0, True,
                 f"{s['ecom']} order #{s['parcel']} delivered. Cash on delivery Tk {s['amt']} paid. Thank you."))

    # 4. Legit Security Alert / Scam Warning / Bill Reminder / Real Cashback Confirmation
    rows.append(("c0_sec_alert", "english", 0, "NONE", 0, True,
                 f"Security alert: New device login on {s['mfs']}. If this was you, ignore. We never ask for your PIN or OTP."))
    rows.append(("c0_sec_alert", "banglish", 0, "NONE", 0, True,
                 f"Sabdhan! Keu jodi {s['mfs']} agent bole PIN ba OTP chay, kokhono diben na. Eta scam."))
    rows.append(("c0_bill_remind", "mixed", 0, "NONE", 0, True,
                 f"Reminder: Your {s['telco']} bill of Tk {s['amt']} is due on 3 Dec. Pay via official app or {s['lurl']}"))
    rows.append(("c0_cashback_real", "bangla", 0, "NONE", 0, True,
                 f"আপনার {s['ecom']} পেমেন্টের জন্য Tk {s['amt']} ক্যাশব্যাক আপনার {s['mfs']} অ্যাকাউন্টে যোগ হয়েছে।"))

    # 5. Ordinary Personal Chat / Appointment
    rows.append(("c0_personal", "english", 0, "NONE", 0, False,
                 f"Hi {s['name']}, meeting moved to 3 PM. Please join the calendar invite link."))
    rows.append(("c0_personal", "bangla", 0, "NONE", 0, False,
                 f"{s['name']}, কাল সকাল ১০টায় ক্লাস হবে। স্যার ইমেইল করেছেন।"))
    rows.append(("c0_personal", "banglish", 0, "NONE", 0, False,
                 f"Mama, ajke raate bari asbo. Bazar theke kichu lagbe? {s['name']} ke bolo."))
    rows.append(("c0_personal", "mixed", 0, "NONE", 0, False,
                 f"Ami ekhon {s['mfs']} e {s['amt']} taka pathaisi, dekh to ashche kina."))

    # ==================== CLASS 1: MFS / OTP SCAM ====================
    # 1. Account lock / KYC threat asking for PIN/OTP/NID
    rows.append(("c1_kyc_lock", "english", 1, "T1|T2|T5", 1, True,
                 f"{s['mfs']} Helpline: Unusual login detected. Reply with your 6-digit OTP and PIN within 1 hour or account will be blocked."))
    rows.append(("c1_kyc_lock", "bangla", 1, "T1|T2|T5", 1, True,
                 f"প্রিয় গ্রাহক, আপনার {s['mfs']} অ্যাকাউন্ট ২৪ ঘণ্টার মধ্যে বন্ধ হয়ে যাবে। চালু রাখতে পিন ও ওটিপি নম্বর পাঠান।"))
    rows.append(("c1_kyc_lock", "banglish", 1, "T1|T2|T5", 1, True,
                 f"Dear {s['mfs']} user, KYC update na korle account band hobe. Update korte NID number o PIN pathan."))
    rows.append(("c1_kyc_lock", "mixed", 1, "T1|T2|T5", 1, True,
                 f"{s['bank']} Security: Card ending {s['otp'][:4]} flagged. Share the OTP you just received. Last warning."))

    # 2. Mistaken transfer claim asking for money back / PIN
    rows.append(("c1_mistake_tx", "banglish", 1, "T1|T2|T5", 1, True,
                 f"Ami {s['mfs']} agent, bhul number-e {s['amt']} taka pathaisi. Ferot din, PIN ta bolen."))
    rows.append(("c1_mistake_tx", "mixed", 1, "T3|T5", 2, True,
                 f"Bhai ekta bhul hoye geche. Amar {s['amt']} taka vul kore apnar {s['mfs']} e chole gese. Please ferot den {s['phone']} e. Ami bipode."))
    rows.append(("c1_mistake_tx", "bangla", 1, "T1|T2|T5", 1, True,
                 f"আমি {s['mfs']} এজেন্ট, ভুল করে আপনার নম্বরে {s['amt']} টাকা গেছে। টাকা ফেরত দিতে ওটিপি কোডটি বলুন।"))
    rows.append(("c1_mistake_tx", "english", 1, "T1|T3|T5", 2, True,
                 f"Mistaken transfer of Tk {s['amt']} sent to your {s['mfs']}. Return immediately to {s['phone']} or police action will be taken."))

    # 3. Card / Refund team phishing
    rows.append(("c1_card_refund", "english", 1, "T1|T2", 1, False,
                 f"{s['ecom']} refund team: send card number, expiry and CVV to receive your Tk {s['amt']} refund."))
    rows.append(("c1_card_refund", "banglish", 1, "T1|T2", 1, False,
                 f"{s['mfs']} bonus: apni Tk {s['amt']} peyechen. Wallet e add korte PIN o card CVV din."))

    # ==================== CLASS 2: PRIZE / LOTTERY SCAM ====================
    rows.append(("c2_lottery_link", "english", 2, "T4|T5", 3, False,
                 f"Congratulations! You won Tk {s['amt']} cashback. Click {s['purl']} to claim within 1 hour."))
    rows.append(("c2_lottery_link", "bangla", 2, "T4|T5", 3, False,
                 f"অভিনন্দন! আপনি {s['amt']} টাকা লটারি জিতেছেন। পুরস্কার পেতে এখনই লিংকে ক্লিক করুন: {s['purl']}"))
    rows.append(("c2_lottery_link", "banglish", 2, "T4|T5", 3, False,
                 f"Apni {s['amt']} taka jitechen! Claim korte link e click korun: {s['purl']}"))
    rows.append(("c2_lottery_link", "mixed", 2, "T1|T4|T5", 3, True,
                 f"Free 10GB data gift from {s['telco']} for you! Tap {s['purl']} now, offer expires in 1 hour."))

    rows.append(("c2_prize_fee", "english", 2, "T1|T3|T4", 2, False,
                 f"You won Tk {s['amt']} from {s['telco']} lucky draw! Pay Tk 300 processing fee via {s['mfs']} to claim."))
    rows.append(("c2_prize_fee", "banglish", 2, "T1|T4", 5, False,
                 f"Congratulations! Apnar number ti {s['telco']} Eid lucky draw te select hoyeche. iPhone jitar jonno {s['phone']} e WhatsApp korun."))
    rows.append(("c2_prize_app", "mixed", 2, "T1|T4", 4, False,
                 f"{s['mfs']} anniversary: spin and win Tk {s['amt']}! Download the spin app: {s['purl']}/app.apk"))
    rows.append(("c2_prize_details", "english", 2, "T4", 1, False,
                 f"Lucky draw winner! Reply with your full name, NID and address to receive a brand new car."))

    # ==================== CLASS 3: FAKE JOB / INVESTMENT ====================
    rows.append(("c3_task_earn", "english", 3, "T6", 4, False,
                 f"Part-time job: like YouTube videos, earn Tk {s['amt']} per task. Join Telegram: {s['purl']}"))
    rows.append(("c3_task_earn", "bangla", 3, "T6", 4, False,
                 f"ঘরে বসে প্রতিদিন {s['amt']} টাকা আয় করুন। সহজ লাইক ও কমেন্টের কাজ। আজই হোয়াটসঅ্যাপে যোগ দিন।"))
    rows.append(("c3_task_earn", "banglish", 3, "T6", 4, False,
                 f"Ghor bose din e {s['amt']} taka income. Facebook page like korle taka. WhatsApp e join korun."))
    rows.append(("c3_task_earn", "mixed", 3, "T6", 4, False,
                 f"Task: buy products on {s['ecom']} via our link, we refund + 20% commission. Complete 5 tasks, get Tk {s['amt']}."))

    rows.append(("c3_invest_fee", "english", 3, "T3|T6", 2, False,
                 f"Double your money in 7 days! Invest Tk {s['amt']}, get double guaranteed. Send via {s['mfs']} {s['phone']}."))
    rows.append(("c3_invest_fee", "mixed", 3, "T1|T3|T6", 2, False,
                 f"Congratulations, selected for Data Entry Officer at {s['emp']}. Pay Tk {s['amt']} registration fee to confirm."))
    rows.append(("c3_crypto", "english", 3, "T5|T6", 4, False,
                 f"{s['crypto']} VIP signals group - 300% weekly profit guaranteed! Join now: {s['purl']}"))
    rows.append(("c3_abroad_job", "banglish", 3, "T3|T6", 5, False,
                 f"Abroad job in Malaysia, visa fee Tk {s['amt']}. 100% guaranteed. Call {s['phone']} ekhoni."))

    # ==================== CLASS 4: DELIVERY / PARCEL PHISHING ====================
    rows.append(("c4_parcel_fee", "english", 4, "T1|T3", 2, True,
                 f"{s['courier']}: Parcel {s['parcel']} held at hub. Pay Tk {s['amt']} storage fee: {s['purl']}"))
    rows.append(("c4_parcel_fee", "bangla", 4, "T3", 2, True,
                 f"আপনার পার্সেল {s['parcel']} আটকে আছে। ডেলিভারি চার্জ {s['amt']} টাকা পরিশোধ করতে লিংকে ক্লিক করুন: {s['purl']}"))
    rows.append(("c4_parcel_fee", "banglish", 4, "T3", 2, True,
                 f"Parcel {s['parcel']} atke ase. Delivery fee Tk {s['amt']} dite link e jan: {s['purl']}"))
    rows.append(("c4_customs", "mixed", 4, "T1|T3", 2, True,
                 f"Customs hold: your parcel from China needs Tk {s['amt']} duty. Pay via {s['mfs']} {s['phone']} to release."))

    rows.append(("c4_addr_link", "english", 4, "T1|T5", 3, True,
                 f"Courier notice: package #{s['parcel']} requires address confirmation. Click {s['purl']} within 24h."))
    rows.append(("c4_addr_link", "bangla", 4, "T1|T5", 3, True,
                 f"{s['courier']}: ডেলিভারি ব্যর্থ হয়েছে। ঠিকানা আপডেট করতে ১২ ঘণ্টার মধ্যে {s['purl']} এ যান।"))
    rows.append(("c4_addr_call", "banglish", 4, "T1|T5", 5, True,
                 f"{s['courier']}: Apnar parcel er address vul. Call korun {s['phone']} ekhoni."))

    # ==================== CLASS 5: PROMOTIONAL SPAM ====================
    rows.append(("c5_promo_offer", "english", 5, "NONE", 0, True,
                 f"Get 10% cashback on your next top-up with code EID10. Valid today only!"))
    rows.append(("c5_promo_offer", "bangla", 5, "NONE", 0, True,
                 f"ঈদ উপলক্ষে {s['ecom']}-এ সব পণ্যে ৩০% ছাড়! আজই কিনুন, শর্ত প্রযোজ্য।"))
    rows.append(("c5_promo_offer", "banglish", 5, "NONE", 0, True,
                 f"Eid offer! {s['ecom']} e sob item e 30% discount. Shop now: {s['lurl']}"))
    rows.append(("c5_promo_offer", "mixed", 5, "NONE", 0, True,
                 f"{s['telco']} Mega Offer! Buy 1GB get 1GB free for Tk {s['amt']}. Dial *121*3#"))
    rows.append(("c5_promo_app", "english", 5, "NONE", 0, False,
                 f"Install {s['ecom']} app and get Tk 50 off your first order. Use code WELCOME50."))
    rows.append(("c5_promo_biz", "mixed", 5, "NONE", 0, False,
                 f"Book your dream apartment in Bashundhara, easy installment. Call {s['phone']}"))
    rows.append(("c5_promo_course", "english", 5, "NONE", 0, False,
                 f"Learn English in 30 days! Admission going on at {s['emp']}. Call now."))

    return rows


def _gen_multiturn_templates(rng: random.Random, s: dict) -> list:
    """Generate multi-turn conversations (2-12 turns) with delayed intent, small talk, or strategy shifts."""
    rows = []
    spk_s, spk_r = rng.choice([("S", "R"), ("Caller", "You"), ("Agent", "Customer"), ("A", "B")])

    # Class 0 multi-turn: personal chat or legit support
    t_c0_1 = [
        f"{spk_s}: Bhai, kemon asos?",
        f"{spk_r}: Alhamdulillah bhalo. Tui?",
        f"{spk_s}: Bhalo. {s['amt']} taka lagbe, Sunday ferot dibo?",
        f"{spk_r}: Ha thik ase, {s['mfs']} e pathacchi.",
        f"{spk_s}: Ok thanks!"
    ]
    rows.append(("c0_mt_chat", "banglish", 0, "NONE", 0, True, len(t_c0_1), " ||| ".join(t_c0_1)))

    t_c0_2 = [
        f"{spk_r}: I forgot my {s['mfs']} PIN. How can I reset it?",
        f"{spk_s}: Dial *247# and choose Reset PIN.",
        f"{spk_s}: Never share your PIN or OTP with anyone, including our support staff.",
        f"{spk_r}: Thank you!"
    ]
    rows.append(("c0_mt_support", "english", 0, "NONE", 0, True, len(t_c0_2), " ||| ".join(t_c0_2)))

    t_c0_3 = [
        f"{spk_r}: আমার পার্সেল {s['parcel']} কি আজ আসবে?",
        f"{spk_s}: হ্যাঁ, রাইডার বিকেল ৫টার দিকে ডেলিভারি করবে।",
        f"{spk_r}: ঠিক আছে, আমি ক্যাশ অন ডেলিভারি দেবো।"
    ]
    rows.append(("c0_mt_delivery", "bangla", 0, "NONE", 0, True, len(t_c0_3), " ||| ".join(t_c0_3)))

    # Class 1 multi-turn: trust building -> mistaken transfer -> PIN/OTP ask (up to 7 turns)
    t_c1_1 = [
        f"{spk_s}: Assalamu alaikum, ami {s['mfs']} office theke bolchi.",
        f"{spk_r}: Wa alaikumussalam, bolun.",
        f"{spk_s}: Apnar account e technical check cholche. Bhul kore {s['amt']} taka chole gese.",
        f"{spk_r}: Ami to kono SMS pai nai.",
        f"{spk_s}: Wait korun, ekta 6 digit code asbe.",
        f"{spk_s}: Account verify korte oi OTP ar apnar PIN ta bolen, na hole account block hobe."
    ]
    rows.append(("c1_mt_agent", "banglish", 1, "T1|T2|T5", 1, True, len(t_c1_1), " ||| ".join(t_c1_1)))

    t_c1_2 = [
        f"{spk_s}: Hello, I am calling from {s['bank']} security.",
        f"{spk_r}: Yes, is there an issue?",
        f"{spk_s}: Someone tried to withdraw Tk {s['amt']} from your card.",
        f"{spk_r}: Oh no! What should I do?",
        f"{spk_s}: To cancel it, quickly tell me the OTP you just received and your card CVV."
    ]
    rows.append(("c1_mt_bank", "english", 1, "T1|T2|T5", 1, True, len(t_c1_2), " ||| ".join(t_c1_2)))

    # Class 2 multi-turn: prize bait -> fee or link
    t_c2_1 = [
        f"{spk_s}: Hello {s['name']}, you have been selected for our {s['ecom']} loyalty reward!",
        f"{spk_r}: Really? What reward?",
        f"{spk_s}: A free smartphone worth 45000 BDT.",
        f"{spk_r}: How do I get it?",
        f"{spk_s}: Just pay Tk {s['amt']} delivery charge via {s['mfs']} to {s['phone']}."
    ]
    rows.append(("c2_mt_prize", "english", 2, "T1|T3|T4", 2, False, len(t_c2_1), " ||| ".join(t_c2_1)))

    t_c2_2 = [
        f"{spk_s}: অভিনন্দন! আপনি {s['telco']} লটারিতে {s['amt']} টাকা জিতেছেন।",
        f"{spk_r}: সত্যি নাকি? কীভাবে পাবো?",
        f"{spk_s}: এই লিংকে গিয়ে ফর্ম পূরণ করুন: {s['purl']}। ১ ঘণ্টার মধ্যে ক্লেইম করুন।"
    ]
    rows.append(("c2_mt_lottery_bn", "bangla", 2, "T1|T4|T5", 3, False, len(t_c2_2), " ||| ".join(t_c2_2)))

    t_c4_2 = [
        f"{spk_s}: আপনার {s['courier']} পার্সেল #{s['parcel']} কাস্টমসে আটকে আছে।",
        f"{spk_r}: আমি তো কিছু অর্ডার করিনি।",
        f"{spk_s}: আপনার নামে এসেছে। ছাড়াতে {s['amt']} টাকা ফি দিন: {s['purl']}"
    ]
    rows.append(("c4_mt_customs_bn", "bangla", 4, "T1|T3|T5", 2, True, len(t_c4_2), " ||| ".join(t_c4_2)))

    t_c5_2 = [
        f"{spk_s}: {s['ecom']} মেগা সেলে আজ সব পণ্যে ৪০% ছাড়!",
        f"{spk_s}: অফার নিতে ভিজিট করুন {s['lurl']}"
    ]
    rows.append(("c5_mt_promo_bn", "bangla", 5, "NONE", 0, True, len(t_c5_2), " ||| ".join(t_c5_2)))

    t_c1_3 = [
        f"{spk_s}: হ্যালো, আমি {s['mfs']} কাস্টমার কেয়ার থেকে বলছি।",
        f"{spk_r}: জ্বী বলুন।",
        f"{spk_s}: আপনার একাউন্টে সমস্যা হয়েছে, দ্রুত আপনার পিন ও ওটিপি দিন।"
    ]
    rows.append(("c1_mt_agent_bn", "bangla", 1, "T1|T2|T5", 1, True, len(t_c1_3), " ||| ".join(t_c1_3)))

    t_c0_4 = [
        f"{spk_r}: Bhai, {s['ecom']} order #{s['parcel']} ki ashche?",
        f"{spk_s}: Ha, rider call korse, 5 tay delivery hobe.",
        f"{spk_r}: Ok, cash on delivery dibo."
    ]
    rows.append(("c0_mt_delivery_mix", "mixed", 0, "NONE", 0, True, len(t_c0_4), " ||| ".join(t_c0_4)))

    t_c4_3 = [
        f"{spk_s}: {s['courier']}: Parcel #{s['parcel']} address incomplete.",
        f"{spk_r}: Ki korte hobe?",
        f"{spk_s}: Confirm address within 12h at {s['purl']} or parcel return hobe."
    ]
    rows.append(("c4_mt_addr_mix", "mixed", 4, "T1|T5", 3, True, len(t_c4_3), " ||| ".join(t_c4_3)))

    # Class 3 multi-turn: job interview -> strategy shift to registration fee / Telegram task
    t_c3_1 = [
        f"{spk_s}: Hi, I saw your CV on {s['emp']}. We have a work-from-home position.",
        f"{spk_r}: What is the job and salary?",
        f"{spk_s}: Data entry, 35,000 BDT per month.",
        f"{spk_r}: Sounds good, how do I start?",
        f"{spk_s}: You need to deposit Tk {s['amt']} as security/registration fee first."
    ]
    rows.append(("c3_mt_jobfee", "english", 3, "T1|T3|T6", 2, False, len(t_c3_1), " ||| ".join(t_c3_1)))

    t_c3_2 = [
        f"{spk_s}: Hello, part-time online kaj korte chan?",
        f"{spk_r}: Ki kaj?",
        f"{spk_s}: YouTube video like korle daily {s['amt']} taka income.",
        f"{spk_s}: First e Telegram group e join korun: {s['purl']}"
    ]
    rows.append(("c3_mt_task", "mixed", 3, "T6", 4, False, len(t_c3_2), " ||| ".join(t_c3_2)))

    # Class 4 multi-turn: parcel notice -> customs fee or address link
    t_c4_1 = [
        f"{spk_s}: Hello, this is {s['courier']}. You have a package #{s['parcel']} from abroad.",
        f"{spk_r}: I didn't order anything.",
        f"{spk_s}: It is registered under your name at customs.",
        f"{spk_r}: What do I need to do?",
        f"{spk_s}: Pay Tk {s['amt']} customs duty today or it will be returned."
    ]
    rows.append(("c4_mt_customs", "english", 4, "T1|T3|T5", 2, True, len(t_c4_1), " ||| ".join(t_c4_1)))

    # Class 5 multi-turn: promo dialogue
    t_c5_1 = [
        f"{spk_s}: Sir, hair fall problem? Our herbal oil is very popular.",
        f"{spk_s}: 40% discount today only, order now at {s['lurl']}."
    ]
    rows.append(("c5_mt_promo", "english", 5, "NONE", 0, True, len(t_c5_1), " ||| ".join(t_c5_1)))

    # Longer 8-10 turn conversation (Class 1 or Class 3) to cover up to 12 turns
    t_long = [
        f"{spk_s}: Assalamu alaikum, apnar sathe kotha bola jabe?",
        f"{spk_r}: Ji bolun, ke bolchen?",
        f"{spk_s}: Ami {s['emp']} HR theke {s['name']}.",
        f"{spk_r}: Oh accha, ki bishoye?",
        f"{spk_s}: Apni amader online task program e select hoyechen.",
        f"{spk_r}: Eta ki rokom kaj?",
        f"{spk_s}: Ghor bose daily {s['amt']} taka income kora jabe.",
        f"{spk_r}: Kono fee lagbe?",
        f"{spk_s}: Ji, shudhu {s['amt']} taka registration fee diye WhatsApp group e join korte hobe."
    ]
    rows.append(("c3_mt_long", "banglish", 3, "T1|T3|T6", 4, False, len(t_long), " ||| ".join(t_long)))

    return rows


def generate_synthetic_dataset(target_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    # Target counts per class (extra weight for 0 and 5 as requested)
    target_per_class = {
        0: int(target_rows * 0.20),  # 1200
        1: int(target_rows * 0.155), # 930
        2: int(target_rows * 0.155), # 930
        3: int(target_rows * 0.155), # 930
        4: int(target_rows * 0.155), # 930
        5: int(target_rows * 0.18),  # 1080
    }

    buckets = {k: [] for k in range(6)}

    # Keep generating batches until all class buckets are full
    while any(len(buckets[k]) < target_per_class[k] for k in range(6)):
        slots = _build_slots(rng)
        single_rows = _gen_single_templates(slots)
        mt_rows = _gen_multiturn_templates(rng, slots)

        # ~65% single, ~35% multi-turn weighting, boosting non-English slightly for ~25% language balance
        for fam, lang, lbl, tech, act, hard, txt in single_rows:
            lang_p = 0.40 if lang == "english" else 0.72
            if len(buckets[lbl]) < target_per_class[lbl] and rng.random() < lang_p:
                buckets[lbl].append({
                    "text": txt,
                    "label": lbl,
                    "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
                    "action": act,
                    "family_id": fam,
                    "language": lang,
                    "n_turns": 1,
                    "is_hard_negative": hard,
                })

        for fam, lang, lbl, tech, act, hard, n_t, txt in mt_rows:
            if len(buckets[lbl]) < target_per_class[lbl] and rng.random() < 0.85:
                buckets[lbl].append({
                    "text": txt,
                    "label": lbl,
                    "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
                    "action": act,
                    "family_id": fam,
                    "language": lang,
                    "n_turns": n_t,
                    "is_hard_negative": hard,
                })

    all_rows = []
    for k in range(6):
        all_rows.extend(buckets[k][:target_per_class[k]])

    rng.shuffle(all_rows)
    df = pd.DataFrame(all_rows)
    df.insert(0, "id", range(100001, 100001 + len(df)))
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(target_rows=6000, seed=42)
    print("Generated shape:", df.shape)
    print("Labels:\n", df["label"].value_counts().sort_index())
    print("Languages:\n", df["language"].value_counts())
    print("Multi-turn ratio:", (df["n_turns"] > 1).mean())
    print("Hard negative ratio:", df["is_hard_negative"].mean())
