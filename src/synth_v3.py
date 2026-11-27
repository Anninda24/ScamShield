import os
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text
from noise import apply_noise

SEED = 42

MFS_NAMES = ["bKash", "Nagad", "Rocket", "Upay", "Cellfin", "SureCash", "Tap", "PayBondhu", "TakaSwift", "DeshPay"]
BANK_NAMES = ["DBBL", "City Bank", "BRAC Bank", "Islami Bank", "EBL", "UCB", "Prime Bank", "NodiBank", "PadmaTrust Bank"]
COURIER_NAMES = ["Sundarban", "SA Paribahan", "Pathao", "RedX", "Paperfly", "Steadfast", "Karatoa", "QuickParcel", "SonarCourier"]
ECOM_NAMES = ["Daraz", "Chaldal", "Aarong", "Shwapno", "Pickaboo", "Rokomari", "MeghnaMart", "ShebaHaat"]
TELCO_NAMES = ["GP", "Robi", "Banglalink", "Teletalk", "PadmaTel"]
MERCHANTS = ["Shwapno", "Meena Bazar", "Unimart", "Aarong", "Pathao Food", "Foodpanda", "Star Cineplex", "Lazz Pharma"]
HUBS = ["Mirpur", "Uttara", "Dhanmondi", "Motijheel", "Chattogram", "Sylhet", "Rajshahi", "Khulna", "Bogura", "Cumilla"]
NAMES = ["Tariq", "Nadia", "Rafiq", "Mim", "Sajib", "Farhana", "Kamal", "Tania", "Imran", "Sadia"]
BAD_DOMAINS = ["bd-verify[.]top", "parcel-pay[.]xyz", "mfs-claim[.]site", "fee-clear[.]vip", "addr-fix[.]top", "track-bd[.]info"]
GOOD_DOMAINS = ["mygp[.]grameenphone[.]com", "daraz[.]com[.]bd/orders", "bracbank[.]com/alerts", "desco[.]org[.]bd/bill", "sundarbancourier[.]com[.]bd/hub"]


def _slots(rng: random.Random) -> dict:
    return {
        "mfs": rng.choice(MFS_NAMES),
        "bank": rng.choice(BANK_NAMES),
        "courier": rng.choice(COURIER_NAMES),
        "ecom": rng.choice(ECOM_NAMES),
        "telco": rng.choice(TELCO_NAMES),
        "merch": rng.choice(MERCHANTS),
        "hub": rng.choice(HUBS),
        "name": rng.choice(NAMES),
        "amt": rng.choice(["60", "85", "120", "350", "450", "750", "1,100", "1,450", "2,200", "3,800", "4,500", "7,500", "12,000"]),
        "bal": rng.choice(["1,340", "2,890", "4,120", "8,750", "14,200", "23,410", "31,050"]),
        "otp": str(rng.randint(100000, 999999)),
        "pid": f"{rng.choice(['PK', 'DX', 'BD', 'TR', 'CR'])}{rng.randint(10000, 99999)}",
        "trx": "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7)),
        "phone": f"{rng.choice(['017', '018', '019', '016', '013'])}XXXXXX{rng.randint(10, 99)}",
        "burl": f"hxxps://{rng.choice(BAD_DOMAINS)}/{rng.choice(['confirm', 'pay', 'verify', 'claim'])}",
        "gurl": f"https://{rng.choice(GOOD_DOMAINS)}",
    }


def _mt(rng: random.Random, turns: list) -> tuple:
    spk_a, spk_b = rng.choice([("S", "R"), ("A", "B")])
    out = [f"{spk_a if i % 2 == 0 else spk_b}: {t}" for i, t in enumerate(turns)]
    return len(out), " ||| ".join(out)


def _gen_c1(rng: random.Random, s: dict) -> list:
    out = []
    # 1. Family-emergency / casual Banglish & Bangla PIN requests
    out.append(("1_v3_family_emerg", "banglish", 1, "T2|T5", 1, True, 1,
                f"Vai hospital e atke gesi, {s['mfs']} e taka nai. Tor PIN ta dao, ekhoni bill diye ferot dibo."))
    out.append(("1_v3_family_emerg", "bangla", 1, "T2|T5", 1, True, 1,
                f"দোস্ত রাস্তায় গাড়ি নষ্ট হয়ে বিপদে আছি, তোমার {s['mfs']} পিনটা দাও তো, জরুরি টাকা তুলতে হবে।"))
    out.append(("1_v3_family_emerg", "mixed", 1, "T2|T5", 1, True, 1,
                f"Mama urgent medical bill dite hobe, amar {s['mfs']} kaj kortese na. Apnar PIN ar OTP ta den."))
    n_t, mt = _mt(rng, [
        f"Kire {s['name']}, koi asos?",
        "Basay vai, bol.",
        "Ekta boro bipode porchi, bus counter e taka shesh.",
        "Koto lagbe?",
        f"Amar account lock, tor {s['mfs']} PIN ta dao ami ekhon nijei send kore nibo."
    ])
    out.append(("1_v3_family_emerg", "banglish", 1, "T2|T5", 1, True, n_t, mt))

    # 2. Varied mistaken-transfer claims (both PIN-ask and money-back)
    out.append(("1_v3_mistaken_v3", "banglish", 1, "T1|T2|T5", 1, True, 1,
                f"{s['mfs']} office: Apnar wallet e galti kore Tk {s['amt']} dhukse. Reverse korte 6 digit code ta bolen."))
    out.append(("1_v3_mistaken_v3", "bangla", 1, "T3|T5", 2, True, 1,
                f"ভাই ভুল করে আপনার {s['mfs']} নম্বরে {s['amt']} টাকা চলে গেছে, আমি খুব সমস্যায় আছি। দয়া করে {s['phone']} এ ফেরত দিন।"))
    out.append(("1_v3_mistaken_v3", "english", 1, "T1|T2|T5", 1, True, 1,
                f"Wrong dispatch alert: Tk {s['amt']} credited to your {s['mfs']} by mistake. Reply with your OTP to cancel."))
    n_t, mt = _mt(rng, [
        f"Salam vai, ami {s['mfs']} agent point theke {s['name']}.",
        "Ji bolun.",
        f"Ek customer er {s['amt']} taka bhul kore apnar number e gese.",
        "Amar kache to kono message ashe nai.",
        "Network delay hocche, ekta code jabe oita ar PIN ta bolle ami pull back kore nibo."
    ])
    out.append(("1_v3_mistaken_v3", "mixed", 1, "T1|T2|T5", 1, True, n_t, mt))

    # 3. Reply-with-code alerts & fake support calls
    out.append(("1_v3_reply_code", "mixed", 1, "T1|T2|T5", 1, True, 1,
                f"{s['bank']} Alert: Suspicious login from Dhaka. Block korte 6 digit code reply korun."))
    out.append(("1_v3_reply_code", "banglish", 1, "T1|T2|T5", 1, True, 1,
                f"Apnar {s['mfs']} app theke onno device login chesta hocche. Code ta bolen ekhoni bondho korchi."))
    out.append(("1_v3_reply_code", "english", 1, "T1|T2|T5", 1, True, 1,
                f"{s['mfs']} Fraud Desk: Someone initiated a Tk {s['amt']} cash-out. Reply with the OTP to stop it."))
    n_t, mt = _mt(rng, [
        f"Hello, {s['bank']} card security theke bolchi.",
        "Ji, kono somossa?",
        f"Apnar card theke Tk {s['amt']} online payment request asche.",
        "Ami to kori nai!",
        "Tahole cancel korte phone e asha 6 digit OTP ar card er last 4 digit bolun."
    ])
    out.append(("1_v3_support_call", "mixed", 1, "T1|T2|T5", 1, True, n_t, mt))

    # 4. KYC & Wallet-limit upgrade
    out.append(("1_v3_kyc_limit", "banglish", 1, "T1|T2|T5", 1, True, 1,
                f"{s['mfs']} update: Daily limit Tk 50,000 barate NID number ar current PIN ta pathan."))
    out.append(("1_v3_kyc_limit", "bangla", 1, "T1|T2|T5", 1, True, 1,
                f"{s['mfs']} কেওয়াইসি আপডেট না করলে আজ রাতের পর লেনদেন বন্ধ হবে। পিন ও ওটিপি দিন।"))
    out.append(("1_v3_kyc_limit", "english", 1, "T1|T2", 1, True, 1,
                f"{s['mfs']} Wallet Upgrade: Verify your merchant tier by replying with PIN and date of birth."))
    n_t, mt = _mt(rng, [
        f"{s['mfs']} customer care: Apnar account e KYC pending ache.",
        "Ami to agei korsi.",
        "System upgrade er por notun kore korte hoy.",
        "Kivabe korbo?",
        "Shudhu NID number ar apnar 5-digit PIN ta ekhane reply korun."
    ])
    out.append(("1_v3_kyc_limit", "banglish", 1, "T1|T2", 1, True, n_t, mt))
    return out


def _gen_c4(rng: random.Random, s: dict) -> list:
    out = []
    # Short one-line Banglish & Bangla parcel phishing + multi-turn variants
    out.append(("4_v3_short_fee", "banglish", 4, "T3", 2, True, 1,
                f"Parcel atke ase, Tk {s['amt']} delivery fee diye charan: {s['burl']}"))
    out.append(("4_v3_short_fee", "bangla", 4, "T3", 2, True, 1,
                f"পার্সেল {s['pid']} আটকে আছে, {s['amt']} টাকা চার্জ দিতে ক্লিক করুন: {s['burl']}"))
    out.append(("4_v3_short_fee", "banglish", 4, "T1|T3", 2, True, 1,
                f"{s['courier']}: Parcel {s['pid']} hold hoye ase. Tk {s['amt']} storage fee pathan {s['phone']} e."))
    out.append(("4_v3_short_customs", "banglish", 4, "T1|T3|T5", 2, True, 1,
                f"Customs e parcel atke gese, ajkei Tk {s['amt']} duty pay korun: {s['burl']}"))
    out.append(("4_v3_short_customs", "bangla", 4, "T1|T3|T5", 2, True, 1,
                f"কাস্টমস ফি {s['amt']} টাকা পরিশোধ না করলে পার্সেল ফেরত যাবে: {s['burl']}"))
    out.append(("4_v3_short_addr", "banglish", 4, "T1|T5", 3, True, 1,
                f"{s['courier']}: Parcel {s['pid']} er thikana vul. Update korun: {s['burl']}"))
    out.append(("4_v3_short_addr", "bangla", 4, "T1|T5", 3, True, 1,
                f"{s['courier']}: ঠিকানা অসম্পূর্ণ থাকায় ডেলিভারি হয়নি। আপডেট করুন: {s['burl']}"))
    out.append(("4_v3_short_redeliv", "banglish", 4, "T1|T3", 2, True, 1,
                f"Redelivery fee Tk {s['amt']} na dile {s['pid']} cancel hobe: {s['burl']}"))
    n_t, mt = _mt(rng, [
        f"Apnar {s['courier']} parcel {s['pid']} hub e asche.",
        "Kobe delivery hobe?",
        f"Address incomplete, punoray pathate Tk {s['amt']} redelivery fee lagbe {s['burl']} e."
    ])
    out.append(("4_v3_short_redeliv", "banglish", 4, "T1|T3", 2, True, n_t, mt))
    return out


def _gen_c0_hard(rng: random.Random, s: dict) -> list:
    out = []
    # 1. Merchant-payment OTP notices with never-share warnings (twin of Class 1 OTP scams)
    out.append(("0_v3_merch_otp", "english", 0, "NONE", 0, True, 1,
                f"{s['bank']}: Use OTP {s['otp']} for Tk {s['amt']} payment at {s['merch']}. Never disclose this OTP to anyone."))
    out.append(("0_v3_merch_otp", "banglish", 0, "NONE", 0, True, 1,
                f"{s['mfs']} payment OTP {s['otp']} ({s['merch']}, Tk {s['amt']}). Ei code kauke bolben na."))
    out.append(("0_v3_merch_otp", "bangla", 0, "NONE", 0, True, 1,
                f"{s['merch']}-এ {s['amt']} টাকা পেমেন্টের ওটিপি {s['otp']}। এই কোড কাউকে দেবেন না।"))

    # 2. Replies explaining how to reset a PIN that say never share it (twin of 1_v3_support_call)
    n_t, mt = _mt(rng, [
        f"Amar {s['mfs']} PIN bhule gesi, ki korbo?",
        f"App er Forgot PIN option e jan ba *247# dial korun.",
        "Kono agent ke bolte hobe?",
        "Na, nijer PIN kokhono amader staff keo bolben na."
    ])
    out.append(("0_v3_pin_reset_help", "banglish", 0, "NONE", 0, True, n_t, mt))
    n_t, mt = _mt(rng, [
        f"How do I change my {s['bank']} app PIN?",
        "Open Settings > Security > Change PIN inside the official app.",
        "Should I reply with my old PIN here?",
        "No! Never reply with or share your PIN or OTP with anyone."
    ])
    out.append(("0_v3_pin_reset_help", "english", 0, "NONE", 0, True, n_t, mt))

    # 3. Courier hub-arrival & branch-pickup notices (twin of Class 4 parcel fees)
    out.append(("0_v3_hub_pickup", "english", 0, "NONE", 0, True, 1,
                f"{s['courier']}: Parcel {s['pid']} reached {s['hub']} branch. Collect in person with this SMS and photo ID."))
    out.append(("0_v3_hub_pickup", "banglish", 0, "NONE", 0, True, 1,
                f"{s['courier']}: Apnar parcel {s['pid']} {s['hub']} hub e pouchheche. SMS ar NID niye collect korun."))
    out.append(("0_v3_hub_pickup", "bangla", 0, "NONE", 0, True, 1,
                f"{s['courier']}: পার্সেল {s['pid']} {s['hub']} শাখায় পৌঁছেছে। এই মেসেজ ও জাতীয় পরিচয়পত্র দেখিয়ে সংগ্রহ করুন।"))
    out.append(("0_v3_track_legit", "english", 0, "NONE", 0, True, 1,
                f"{s['ecom']} Order {s['pid']} dispatched via {s['courier']}. Track free at {s['gurl']}."))

    # 4. Bank debit alerts & MFS cash-in / cash-out / send-money confirmations
    out.append(("0_v3_bank_debit", "english", 0, "NONE", 0, True, 1,
                f"{s['bank']}: BDT {s['amt']} debited from A/C XX{s['otp'][:4]} at {s['merch']}. Current balance BDT {s['bal']}."))
    out.append(("0_v3_mfs_tx", "banglish", 0, "NONE", 0, True, 1,
                f"{s['mfs']} Cash-In Tk {s['amt']} from {s['phone']} successful. New balance Tk {s['bal']}. Trx {s['trx']}."))
    out.append(("0_v3_mfs_tx", "bangla", 0, "NONE", 0, True, 1,
                f"{s['mfs']}: {s['phone']} নম্বরে {s['amt']} টাকা সেন্ড মানি সম্পন্ন হয়েছে। ব্যালেন্স {s['bal']} টাকা।"))
    out.append(("0_v3_mfs_tx", "mixed", 0, "NONE", 0, True, 1,
                f"Cash-Out Tk {s['amt']} completed at {s['hub']} agent. Fee Tk 5.00. Balance Tk {s['bal']}. TrxID {s['trx']}."))

    # 5. Genuine cashback confirmations, bill reminders, service notices with link
    out.append(("0_v3_real_cashback", "english", 0, "NONE", 0, True, 1,
                f"{s['mfs']}: Tk {s['amt']} instant cashback credited for your {s['merch']} bill payment. Balance Tk {s['bal']}."))
    out.append(("0_v3_bill_link", "mixed", 0, "NONE", 0, True, 1,
                f"WASA water bill for this month is Tk {s['amt']}. Download e-bill at {s['gurl']} or pay via {s['mfs']}."))
    out.append(("0_v3_service_link", "bangla", 0, "NONE", 0, True, 1,
                f"{s['telco']} নোটিশ: নেটওয়ার্ক উন্নয়নের কাজ চলছে। বিস্তারিত জানতে দেখুন: {s['gurl']}"))

    # 6. Multi-turn legit chat / delivery
    n_t, mt = _mt(rng, [
        f"Bhai, {s['merch']} er bill ta share korbo?",
        f"Ha, amar {s['mfs']} e Tk {s['amt']} pathay dis.",
        "Ok pathaisi, check kor.",
        "Paisi, dhonyobad."
    ])
    out.append(("0_v3_friend_tx", "banglish", 0, "NONE", 0, True, n_t, mt))
    return out


def _gen_c5(rng: random.Random, s: dict) -> list:
    out = []
    # Promos mentioning cashback, offers, links, and OTP-free app promos
    out.append(("5_v3_cashback_promo", "english", 5, "NONE", 0, True, 1,
                f"Enjoy 15% cashback up to Tk 150 on {s['merch']} when paying with {s['mfs']} app! Valid till Friday."))
    out.append(("5_v3_cashback_promo", "banglish", 5, "NONE", 0, True, 1,
                f"{s['ecom']} App Offer: Order korlei Tk 100 cashback, kono OTP ba fee lagbe na. Shop: {s['gurl']}"))
    out.append(("5_v3_cashback_promo", "bangla", 5, "NONE", 0, True, 1,
                f"{s['mfs']} অ্যাপ থেকে মোবাইল রিচার্জে ১০% ক্যাশব্যাক অফার! বিস্তারিত: {s['gurl']}"))
    out.append(("5_v3_app_promo", "mixed", 5, "NONE", 0, True, 1,
                f"Download {s['ecom']} app and get flat 20% off first grocery order. Code: FRESH20. {s['gurl']}"))
    n_t, mt = _mt(rng, [
        f"{s['ecom']} Weekend Sale shuru hoyeche!",
        "Ki ki offer ache?",
        f"Sob winter item e 35% discount + {s['mfs']} e 10% cashback. Visit {s['gurl']}."
    ])
    out.append(("5_v3_cashback_promo", "banglish", 5, "NONE", 0, True, n_t, mt))
    return out


def _gen_noise_only(rng: random.Random) -> tuple:
    letters = "abcdefghijklmnopqrstuvwxyz"
    consonants = "bcdfghjklmnpqrstvwxyz"
    n_tok = rng.randint(1, 3)
    toks = []
    for _ in range(n_tok):
        alpha = consonants if rng.random() < 0.6 else letters
        toks.append("".join(rng.choice(alpha) for _ in range(rng.randint(4, 12))))
    txt = " ".join(toks)
    if rng.random() < 0.25:
        txt += rng.choice([".", "!", "?"])
    if rng.random() < 0.15:
        txt = rng.choice(["📲", "⚠️", "🔥"]) + " " + txt
    return ("0_v3_noise_only", "english", 0, "NONE", 0, False, 1, txt)


def generate_v3_patch(seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed + 303)
    real_df = pd.read_csv("sample.csv")
    v2_df = pd.read_csv("data/synthetic_v2.csv", keep_default_na=False)

    # Target counts for new rows (~1,200 total)
    targets = {1: 350, 4: 200, 0: 450, 5: 100}
    buckets = {1: [], 4: [], 0: [], 5: []}

    while any(len(buckets[k]) < targets[k] for k in targets):
        s = _slots(rng)
        for scen, lang, lbl, tech, act, hard, n_t, txt in _gen_c1(rng, s):
            if len(buckets[1]) < targets[1] and (n_t > 1 or rng.random() < 0.55):
                buckets[1].append((scen, lang, lbl, tech, act, hard, n_t, txt))
        for scen, lang, lbl, tech, act, hard, n_t, txt in _gen_c4(rng, s):
            if len(buckets[4]) < targets[4] and (n_t > 1 or rng.random() < 0.65):
                buckets[4].append((scen, lang, lbl, tech, act, hard, n_t, txt))
        for scen, lang, lbl, tech, act, hard, n_t, txt in _gen_c0_hard(rng, s):
            if len(buckets[0]) < targets[0] and (n_t > 1 or rng.random() < 0.55):
                buckets[0].append((scen, lang, lbl, tech, act, hard, n_t, txt))
        for scen, lang, lbl, tech, act, hard, n_t, txt in _gen_c5(rng, s):
            if len(buckets[5]) < targets[5]:
                buckets[5].append((scen, lang, lbl, tech, act, hard, n_t, txt))

    new_rows = []
    for k in [1, 4, 0, 5]:
        for scen, lang, lbl, tech, act, hard, n_t, txt in buckets[k][:targets[k]]:
            # Ensure >= 30% multi-turn (3-8 turns) on non-noise rows
            if n_t == 1 and lbl in (0, 1, 5) and rng.random() < 0.22:
                n_t, txt = _mt(rng, [
                    rng.choice(["Hello, free achen?", "Salam, ekta kotha chilo.", "Hi there!"]),
                    rng.choice(["Ji bolun.", "Ha, bolo.", "Yes?"]),
                    txt,
                ])
            new_rows.append({
                "text": txt,
                "label": lbl,
                "techniques": "|".join(sorted(tech.split("|"))) if tech != "NONE" else "NONE",
                "action": act,
                "family_id": scen,
                "scenario_id": scen,
                "language": lang,
                "n_turns": n_t,
                "is_hard_negative": hard,
                "source": "synthetic_v3",
            })

    # Add ~100 noise-only rows
    for _ in range(100):
        scen, lang, lbl, tech, act, hard, n_t, txt = _gen_noise_only(rng)
        new_rows.append({
            "text": txt, "label": lbl, "techniques": tech, "action": act,
            "family_id": scen, "scenario_id": scen, "language": lang,
            "n_turns": n_t, "is_hard_negative": hard, "source": "synthetic_v3",
        })

    new_df = pd.DataFrame(new_rows)

    # Apply noise to 30% of new rows (skip noise-only rows or keep ~30% overall)
    is_noisy = []
    noisy_txts = []
    for _, r in new_df.iterrows():
        if r["scenario_id"] != "0_v3_noise_only" and rng.random() < 0.33:
            noisy_txts.append(apply_noise(r["text"], rng, level=0.35))
            is_noisy.append(True)
        else:
            noisy_txts.append(r["text"])
            is_noisy.append(False)
    new_df["text"] = noisy_txts
    new_df["is_noisy"] = is_noisy

    # Drop exact duplicates among new rows & against v2
    before_dedup = len(new_df)
    v2_texts_set = set(v2_df["text"].astype(str))
    new_df = new_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    new_df = new_df[~new_df["text"].isin(v2_texts_set)].reset_index(drop=True)
    after_dedup = len(new_df)

    # Similarity check against 86 real rows: drop new rows with sim > 0.85
    real_proc = real_df["text"].fillna("").astype(str).apply(preprocess_text).tolist()
    new_proc = new_df["text"].fillna("").astype(str).apply(preprocess_text).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    X_real = vec.fit_transform(real_proc)
    X_new = vec.transform(new_proc)
    max_sims = cosine_similarity(X_new, X_real).max(axis=1)

    sim_drop_mask = max_sims > 0.85
    dropped_sim = int(sim_drop_mask.sum())
    new_df = new_df[~sim_drop_mask].reset_index(drop=True)
    rem_max_sim = float(max_sims[~sim_drop_mask].max()) if len(new_df) > 0 else 0.0

    # Assign sequential IDs after v2 max ID
    start_id = int(v2_df["id"].max()) + 1
    new_df.insert(0, "id", range(start_id, start_id + len(new_df)))

    # Ensure column order matches v2_df
    new_df = new_df[v2_df.columns.tolist()]
    v3_df = pd.concat([v2_df, new_df], ignore_index=True)

    os.makedirs("data", exist_ok=True)
    v3_df.to_csv("data/synthetic_v3.csv", index=False)

    print("=== PHASE 3C TARGETED DATA PATCH REPORT ===")
    print(f"Generated new rows (raw): {before_dedup} | After dedup: {after_dedup} (dropped {before_dedup - after_dedup})")
    print(f"Dropped due to cosine sim > 0.85 vs 86 real rows: {dropped_sim} | Remaining max sim: {rem_max_sim:.4f}")
    print(f"Final new synthetic_v3 rows: {len(new_df)} | Total synthetic_v3.csv shape: {v3_df.shape}")
    print(f"New rows multi-turn ratio: {(new_df['n_turns'] > 1).mean():.4f} | noisy ratio: {new_df['is_noisy'].mean():.4f}")

    print("\nNew rows class counts:", new_df["label"].value_counts().sort_index().to_dict())
    print("Total synthetic_v3.csv class counts:", v3_df["label"].value_counts().sort_index().to_dict())

    print("\nTechniques & Action distributions per class (new synthetic_v3 rows):")
    for lbl in sorted(new_df["label"].unique()):
        sub = new_df[new_df["label"] == lbl]
        print(f"  Class {lbl} (n={len(sub)}): actions={sub['action'].value_counts().to_dict()} | tech_top5={sub['techniques'].value_counts().head(5).to_dict()}")

    print("\n=== 40 RANDOM NEW ROWS (source == 'synthetic_v3') ===")
    samp = new_df.sample(min(40, len(new_df)), random_state=SEED)
    for i, (_, r) in enumerate(samp.iterrows(), 1):
        print(f"[{i:02d}] L={r['label']} | T={r['techniques']:8s} | A={r['action']} | {r['language']:8s} | turns={r['n_turns']} | noisy={r['is_noisy']} | scen={r['scenario_id']}")
        print(f"     {repr(str(r['text'])[:135])}")


if __name__ == "__main__":
    generate_v3_patch(seed=SEED)
