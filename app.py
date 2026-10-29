
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PhishShift | AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# MODERN DARK + NEON THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #152748 0%, transparent 38%),
        radial-gradient(ellipse at 95% 5%, #211641 0%, transparent 35%),
        #080d19;
    color: #edf4ff;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a2e, #090f1b);
    border-right: 1px solid #263650;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f7ff !important;
}

p, label {
    color: #d4deef;
}

.hero {
    padding: 28px;
    border: 1px solid #304666;
    border-radius: 22px;
    background: linear-gradient(120deg, #122640, #211735);
    margin-bottom: 25px;
}

.hero-label {
    color: #62e7f7;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 3px;
}

.hero-title {
    font-size: 38px;
    font-weight: 700;
    margin-top: 10px;
    color: #ffffff;
}

.hero-description {
    color: #b5c5df;
    margin-top: 8px;
}

.tag {
    display: inline-block;
    padding: 5px 11px;
    margin: 12px 6px 0 0;
    border-radius: 20px;
    border: 1px solid #3b5574;
    background: #152b43;
    color: #8feefa;
    font-size: 12px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #131f34, #101727);
    border: 1px solid #2a3b57;
    border-radius: 16px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #9fb2d0 !important;
}

div[data-testid="stMetricValue"] {
    color: #f5f8ff !important;
}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #40d9e9, #9b87ff);
    color: #080d19;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    filter: brightness(1.1);
}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    background: #0e1728 !important;
    border-color: #2a3b57 !important;
    border-radius: 10px !important;
}

textarea, input {
    color: #f0f5ff !important;
}

hr {
    border-color: #283953 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293b57;
    border-radius: 12px;
    overflow: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SYNTHETIC DEMO DATASET
# =========================================================
@st.cache_data
def create_dataset():

    legitimate_templates = [
        "Your appointment is confirmed for {day} at {time}.",
        "The meeting has been moved to {day}. Check your calendar.",
        "Your order has shipped. Track it in the official app.",
        "The class assignment is available on the college portal.",
        "Your payment receipt is ready in your account.",
        "Reminder: your library book is due on {day}.",
        "The project review starts at {time} in the lab.",
        "Your ticket confirmation is available in the official app.",
        "Please attend the scheduled team meeting at {time}.",
        "Your monthly statement is ready in your account.",
        "The exam timetable is available on the college website.",
        "Your subscription reminder is available in your account.",
        "Your support request was received. Reference number {number}.",
        "The delivery is scheduled for {day}.",
        "Please review the shared document before the meeting.",
        "Your class begins at {time}. Please bring your notes.",
        "The team uploaded the project files to the shared folder.",
        "Your booking details are available in the official application.",
    ]

    suspicious_templates = [
        "Urgent: verify your account immediately to avoid suspension.",
        "Your account is locked. Confirm your password to restore access.",
        "You have won a prize. Submit your personal details to claim it.",
        "Your payment failed. Use this unfamiliar link to update details.",
        "Immediate action required: your account will be disabled today.",
        "Confirm your bank details now to prevent account restrictions.",
        "Your parcel is on hold. Pay an unexpected fee to continue.",
        "Security alert: provide your login code to verify identity.",
        "Enter your card details to receive your unexpected refund.",
        "Your account has unusual activity. Verify your credentials now.",
        "Claim your reward by submitting confidential information.",
        "Your access expires soon. Sign in through this unverified link.",
        "Share your one-time code to fix an account problem.",
        "Make an urgent payment to avoid an unexpected penalty.",
        "Provide your password immediately to restore account access.",
        "Your package is delayed. Pay now through this shortened link.",
        "Your profile will be blocked unless you send the login code.",
        "Click now to claim your exclusive reward.",
    ]

    rows = []

    for i in range(240):
        template = legitimate_templates[i % len(legitimate_templates)]

        message = template.format(
            day=["Monday", "Tuesday", "Friday", "tomorrow"][i % 4],
            time=["9:00 AM", "2:30 PM", "4:00 PM"][i % 3],
            number=1000 + i,
        )

        rows.append({
            "message": message,
            "label": 0,
            "category": "Legitimate",
            "channel": ["Email", "SMS", "Chat"][i % 3],
        })

    for i in range(240):
        rows.append({
            "message": suspicious_templates[i % len(suspicious_templates)],
            "label": 1,
            "category": "Suspicious",
            "channel": ["Email", "SMS", "Chat"][i % 3],
        })

    df = pd.DataFrame(rows)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================================================
# MACHINE LEARNING MODELS
# =========================================================
@st.cache_resource
def train_models():

    df = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    models = {
        "Logistic Regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
            ),
            ("model", MultinomialNB()),
        ]),
    }

    metrics = {}

    for name, model in models.items():

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "FPR": fp / (fp + tn) if (fp + tn) else 0,
        }

    return models, metrics


# =========================================================
# RULE-BASED DETECTION
# =========================================================
RISK_PATTERNS = [
    r"\bverify your account\b",
    r"\bconfirm your password\b",
    r"\baccount is locked\b",
    r"\bshare your (?:one-time|login) code\b",
    r"\burgent\b",
    r"\bclaim your (?:prize|reward)\b",
    r"\bprovide your bank details\b",
    r"\bavoid account suspension\b",
    r"\benter your card details\b",
    r"\bimmediate action required\b",
    r"\bshortened link\b",
    r"\bone-time code\b",
    r"\bpassword\b",
    r"\bpay now\b",
]


def analyze_message(message, models):

    cleaned = message.strip()

    if not cleaned:
        return None

    matches = [
        pattern
        for pattern in RISK_PATTERNS
        if re.search(pattern, cleaned, re.IGNORECASE)
    ]

    rule_score = min(len(matches) / 4, 1.0)

    lr_score = float(
        models["Logistic Regression"]
        .predict_proba([cleaned])[0][1]
    )

    nb_score = float(
        models["Naive Bayes"]
        .predict_proba([cleaned])[0][1]
    )

    combined_score = (
        0.40 * lr_score
        + 0.35 * nb_score
        + 0.25 * rule_score
    )

    if len(cleaned.split()) < 3:
        risk = "Uncertain"
    elif combined_score >= 0.70:
        risk = "High"
    elif combined_score >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "message": cleaned,
        "risk": risk,
        "combined_score": combined_score,
        "rule_score": rule_score,
        "rule_matches": len(matches),
        "lr_score": lr_score,
        "nb_score": nb_score,
        "scanned_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


# =========================================================
# LOAD MODELS AND DATA
# =========================================================
models, model_metrics = train_models()
dataset = create_dataset()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:

    st.markdown("## 🛡️ PhishShift")
    st.caption("AI MESSAGE SECURITY CONSOLE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Scan Message",
            "Scan History",
            "Dataset Explorer",
            "Model Evaluation",
        ],
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")
    st.success("Demo models loaded")
    st.caption("TF-IDF feature engine")
    st.caption("Rule indicator engine")
    st.caption("Logistic Regression")
    st.caption("Multinomial Naive Bayes")

    st.divider()
    st.caption("Educational prototype")
    st.caption("Do not enter real passwords or OTPs.")


# =========================================================
# TOP HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">THREAT INTELLIGENCE / MESSAGE ANALYSIS</div>
    <div class="hero-title">
        PhishShift <span style="color:#62e7f7;">Security Console</span>
    </div>
    <div class="hero-description">
        Detect suspicious message patterns with rule-based indicators
        and machine-learning text classification.
    </div>
    <span class="tag">● DEMO SESSION</span>
    <span class="tag">TF-IDF + ML</span>
    <span class="tag">RULE ENGINE</span>
    <span class="tag">RISK ANALYSIS</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "Overview":

    total_scans = len(st.session_state.scan_history)

    high_count = sum(
        item["risk"] == "High"
        for item in st.session_state.scan_history
    )

    medium_count = sum(
        item["risk"] == "Medium"
        for item in st.session_state.scan_history
    )

    low_count = sum(
        item["risk"] == "Low"
        for item in st.session_state.scan_history
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TRAINING EXAMPLES", f"{len(dataset):,}")
    c2.metric("MESSAGES SCANNED", f"{total_scans:,}")
    c3.metric("HIGH RISK", f"{high_count:,}")
    c4.metric("LOW RISK", f"{low_count:,}")

    st.markdown("### Detection Workspace")

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("#### Analysis Pipeline")

        st.markdown("""
        <div style="
            background:#10182b;
            border:1px solid #293b57;
            border-radius:16px;
            padding:22px;
            line-height:2.2;">
            <span style="color:#62e7f7;">01</span> &nbsp; Message Input<br>
            <span style="color:#62e7f7;">02</span> &nbsp; Rule Indicators<br>
            <span style="color:#62e7f7;">03</span> &nbsp; TF-IDF Text Features<br>
            <span style="color:#62e7f7;">04</span> &nbsp; Logistic Regression + Naive Bayes<br>
            <span style="color:#62e7f7;">05</span> &nbsp; Combined Risk Estimate
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("#### Model Snapshot")

        metric_df = pd.DataFrame(model_metrics).T

        st.dataframe(
            metric_df.style.format("{:.1%}"),
            use_container_width=True,
        )

        st.caption(
            "Metrics use synthetic demonstration messages; "
            "they do not establish real-world accuracy."
        )

    st.markdown("### Recent Scan Activity")

    if st.session_state.scan_history:

        history_df = pd.DataFrame(
            st.session_state.scan_history
        ).head(5)

        st.dataframe(
            history_df[
                ["scanned_at", "risk", "combined_score", "message"]
            ].style.format({"combined_score": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No scans yet. Open Scan Message to analyze a message."
        )


# =========================================================
# PAGE 2: SCAN MESSAGE
# =========================================================
elif page == "Scan Message":

    st.markdown("### 🛡️ Message Scanner")

    st.caption(
        "Paste the message to estimate its risk. "
        "Do not enter real passwords, OTPs, or sensitive information."
    )

    with st.form("scan_form"):

        message = st.text_area(
            "MESSAGE CONTENT",
            placeholder=(
                "Example: Urgent, verify your account immediately..."
            ),
            height=170,
        )

        submitted = st.form_submit_button(
            "⚡ ANALYZE MESSAGE",
            use_container_width=True,
        )

    if submitted:

        result = analyze_message(message, models)

        if result is None:

            st.warning("Please enter a message before scanning.")

        else:

            st.session_state.scan_history.insert(0, result)

            st.divider()
            st.markdown("### Scan Verdict")

            if result["risk"] == "High":

                st.error(
                    "HIGH RISK ESTIMATE — inspect the sender "
                    "and verify independently."
                )

            elif result["risk"] == "Medium":

                st.warning(
                    "MEDIUM RISK ESTIMATE — verify through an "
                    "official website or application."
                )

            elif result["risk"] == "Uncertain":

                st.info(
                    "UNCERTAIN — the message is too short "
                    "for a useful estimate."
                )

            else:

                st.success(
                    "LOW RISK ESTIMATE — this does not guarantee "
                    "that the message is safe."
                )

            a, b, c = st.columns(3)

            a.metric("RISK LEVEL", result["risk"])

            b.metric(
                "COMBINED SCORE",
                f"{result['combined_score']:.1%}",
            )

            c.metric(
                "RULE INDICATORS",
                result["rule_matches"],
            )

            st.markdown("#### Engine Score Breakdown")

            score_df = pd.DataFrame({
                "Engine": [
                    "Rule Baseline",
                    "Logistic Regression",
                    "Naive Bayes",
                ],
                "Estimated Score": [
                    result["rule_score"],
                    result["lr_score"],
                    result["nb_score"],
                ],
            }).set_index("Engine")

            st.bar_chart(score_df, horizontal=True)

            st.caption(
                "Scores are estimates from synthetic demo data, "
                "not calibrated real-world probabilities."
            )


# =========================================================
# PAGE 3: SCAN HISTORY
# =========================================================
elif page == "Scan History":

    st.markdown("### 🕘 Session Scan History")

    if st.session_state.scan_history:

        history_df = pd.DataFrame(
            st.session_state.scan_history
        )

        st.dataframe(
            history_df[
                [
                    "scanned_at",
                    "risk",
                    "combined_score",
                    "rule_matches",
                    "message",
                ]
            ].style.format({"combined_score": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ EXPORT HISTORY AS CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="phishshift_scan_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Session History"):

            st.session_state.scan_history = []
            st.rerun()

    else:

        st.info(
            "History is empty. Scan a message to populate this table."
        )


# =========================================================
# PAGE 4: DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":

    st.markdown("### 📁 Dataset Explorer")

    st.caption(
        "This prototype generates 480 examples: 240 legitimate "
        "and 240 suspicious. These are synthetic examples, "
        "not a real collected dataset."
    )

    a, b, c = st.columns(3)

    a.metric("TOTAL EXAMPLES", len(dataset))

    b.metric(
        "LEGITIMATE",
        int((dataset["label"] == 0).sum()),
    )

    c.metric(
        "SUSPICIOUS",
        int((dataset["label"] == 1).sum()),
    )

    selected = st.selectbox(
        "FILTER BY CATEGORY",
        ["All", "Legitimate", "Suspicious"],
    )

    if selected == "All":
        filtered_df = dataset
    else:
        filtered_df = dataset[
            dataset["category"] == selected
        ]

    st.dataframe(
        filtered_df[["message", "category", "channel"]],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ DOWNLOAD DEMO DATASET",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name="phishshift_demo_dataset.csv",
        mime="text/csv",
    )


# =========================================================
# PAGE 5: MODEL EVALUATION
# =========================================================
elif page == "Model Evaluation":

    st.markdown("### 📊 Model Evaluation Lab")

    st.caption(
        "Metrics below are calculated using a held-out split "
        "of the synthetic dataset."
    )

    metrics_df = pd.DataFrame(model_metrics).T

    st.dataframe(
        metrics_df.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("### Metric Guide")

    left, right = st.columns(2)

    with left:

        st.markdown(
            "**Accuracy** — fraction of all predictions "
            "that were correct."
        )

        st.markdown(
            "**Precision** — fraction of flagged messages "
            "that were positive."
        )

    with right:

        st.markdown(
            "**Recall** — fraction of positive messages "
            "that were detected."
        )

        st.markdown(
            "**FPR** — fraction of legitimate messages "
            "incorrectly flagged."
        )

    st.markdown("### Performance Comparison")

    st.bar_chart(
        metrics_df[["Accuracy", "Precision", "Recall", "FPR"]].T
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "PhishShift • AI Security Prototype • "
    "Always verify suspicious messages through official channels. "
    "Not a production security filter."
)
