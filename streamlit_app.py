import re
import joblib
import streamlit as st

from text_cleaner import preprocess_text # still needed for tokenizer consistency
st.set_page_config(page_title="SMS Spam Detector", page_icon="📱", layout="centered")
#Load the trained model and vectorizer

@st.cache_resource
def load_model():
    model = joblib.load("models/spam_svm_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer_adv.pkl")
    return model, vectorizer

svm_clf, vectorizer_adv = load_model()

# Pattern groups with priorities

PATTERN_GROUPS = {
    "financial": {
        "patterns": [
            r"\bearn\b.{0,20}\b\d{2,}\b",              # earn 5000, earn 10000
            r"\bwork\s+from\s+home\b",
            r"\bmake\s+money\b",
            r"\bextra\s+income\b",
            r"\bno\s+investment\b",
        ],
        "priority": 3,   # high
        "min_hits": 1,
    },
    "loan_credit": {
        "patterns": [
            r"\binstant\s+loan\b",
            r"\bpre[- ]?approved\s+loan\b",
            r"\bno\s+document[s]?\b",
            r"\blower\s+interest\b",
        ],
        "priority": 2,   # medium
        "min_hits": 1,
    },
    "phishing": {
        "patterns": [
            r"\byour\s+account\b.{0,20}\b(suspended|blocked|closed)\b",
            r"\bverify\b.{0,15}\b(bank|account|card|details)\b",
            r"\bconfirm\b.{0,15}\b(pin|otp|password)\b",
        ],
        "priority": 3,   # high
        "min_hits": 1,
    },
    "lottery_prize": {
        "patterns": [
            r"\bcongratulation(?:s)?\b.{0,20}\b(won|winner)\b",
            r"\bfree\b\s+(gift|prize|entry|voucher|ticket)\b",
            r"\bselected\b.{0,20}\b(lucky|winner)\b",
        ],
        "priority": 2,   # medium
        "min_hits": 1,
    },
    "generic_promo": {
        "patterns": [
            r"\blimited\s+time\s+offer\b",
            r"\bclick\s+the\s+link\b",
            r"\bcall\s+now\b",
            r"\bact\s+now\b",
        ],
        "priority": 1,   # low
        "min_hits": 2,   # need more than one generic hit
    },
}

def analyze_rule_matches(text: str):
    '''
    Returns:
    -group hits: dict[group_name] = number of pattern matches in that group
    -total_score: sum(priority * hits) over all groups
    -matched_details: list of (group,pattern) for debugging'''
    
    t = text.lower()
    group_hits = {}
    total_score = 0
    matched_details = []
    
    for group_name, cfg in PATTERN_GROUPS.items():
        patterns = cfg['patterns']
        priority = cfg['priority']
        hits_in_group = 0
        for pat in patterns:
            if re.search(pat, t):
                hits_in_group += 1
                matched_details.append((group_name, pat))
        if hits_in_group > 0:
            group_hits[group_name] = hits_in_group
            total_score += priority * hits_in_group
            
    return group_hits, total_score, matched_details

def is_rule_based_spam(text: str):
    '''
    Decide if text is spam based on grouped patterns.
    Returns:
    is_spam: bool
    group_hits: dict of hits per group
    total_score: int'''
    
    group_hits, total_score, matched_details = analyze_rule_matches(text)
    
    for group_name, hits in group_hits.items():
        cfg = PATTERN_GROUPS[group_name]
        priority = cfg["priority"]
        min_hits = cfg["min_hits"]

        if priority >= 3 and hits >= min_hits:
            return True, group_hits, total_score

        if priority == 2 and hits >= min_hits:
            return True, group_hits, total_score

    # Generic promo: require multiple hits or high total score
    if "generic_promo" in group_hits and total_score >= 3:
        return True, group_hits, total_score

    return False, group_hits, total_score


def classify_message(text: str):
    """
    Final classifier:
      1. Run rule engine (grouped patterns).
      2. If rules say spam -> spam (with rule explanation).
      3. Else -> use SVM model.
    Returns:
      label, explanation(dict)
    """
    is_spam_rule, group_hits, total_score = is_rule_based_spam(text)

    if is_spam_rule:
        label = "spam"
        explanation = {
            "via": "rules",
            "group_hits": group_hits,
            "total_score": total_score,
        }
    else:
        vec = vectorizer_adv.transform([text])
        label = svm_clf.predict(vec)[0]
        explanation = {
            "via": "model",
            "group_hits": group_hits,
            "total_score": total_score,
        }

    return label, explanation


# -------------------------
#Streamlit UI
# -------------------------
# st.set_page_config(page_title="SMS Spam Detector",layout='centered',page_icon="📨")

st.title("📱 SMS Spam Detection App")
st.write(
    "Type an SMS message below and the app will predict whether it is **spam** or **ham (not spam)**.\n"
    "The system uses a combination of **Machine Learning (SVM + TF-IDF)** and **rule-based patterns** "
    "for financial, phishing, lottery, and promotional spam."
)

st.markdown("---")

#text input
user_input = st.text_area("Enter SMS message here:",height=150, placeholder="Type your message...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a valid SMS message.")
    else:
        label,explanation = classify_message(user_input)
        if label == "spam":
            st.error("The message is classiied as **SPAM** 🚫")
        else:
            st.success("The message is classified as **HAM (not spam)** ✅")
            
        st.markdown("### Explanation:")
        
        if explanation['via'] == 'rules':
            st.write("The message was classified as **spam** based on matching rule-based patterns.")
           
        else:
            st.write("The message was classified using the **Machine Learning model (SVM + TF-IDF)**.")
           
        group_hits = explanation['group_hits']
        total_score = explanation['total_score']
        
        if group_hits:
            st.write("#### Pattern Group Hits:")
            for group, hits in group_hits.items():
                st.write(f"- **{group}**: {hits} pattern match(es)")
        else:
            st.write("No rule-based pattern matches were found.")
            
        st.write(f"**Total Rule-Based Score:** {total_score}")
        
st.markdown("---")
st.write("Developed by Naman Sethi")

st.subheader("🔍 Try some example messages")

examples = [
    "You have been selected to earn $5000 per week working from home. No investment required. Call now!",
    "Congratulations! You have won a brand new iPhone 15. Click the link to claim your prize.",
    "Your account will be suspended. Verify your bank details immediately.",
    "Hey, are we still meeting for coffee tomorrow?",
    "Mom asked if you could bring some milk on your way back home."
]

example = st.selectbox("Choose an example to paste:", [""] + examples)

if example:
    st.write("click below to copy the example message:")
    if st.button("Use this example"):
        # Streamlit can't directly set text_area value dynamically without session state,
        # so we show it and ask user to copy manually or rerun.
        st.info("Copy the text from below and paste it into the input box above:")
        st.code(example)
        


            
        
