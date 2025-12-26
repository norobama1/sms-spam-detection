# 📱 SMS Spam Detection using NLP, SVM & Streamlit

A production-style **SMS Spam Detection system** built using **Natural Language Processing (NLP)**,
**TF-IDF**, **Support Vector Machine (SVM)**, and a **rule-based filtering layer**.

The project includes a **Streamlit web application** that allows users to classify SMS messages
in real time with an optional **Light/Dark theme toggle** and explainable predictions.

## 🚀 Features

- Advanced text preprocessing (lemmatization, stopword removal using NLTK)
- TF-IDF feature extraction (unigrams + bigrams)
- Linear SVM classifier for text classification
- Rule-based detection for:
  - Financial scams
  - Phishing attempts
  - Lottery / prize spam
- Hybrid decision system (Rules + ML)
- Streamlit web app for real-time prediction
- Light / Dark mode UI toggle
- Explainable predictions (rule-based vs ML-based)

## 🗂 Project Structure

```text
Sms_Detection/
│
├── streamlit_app.py          # Streamlit web app
├── text_cleaner.py           # Text preprocessing & tokenizer
├── sms_detection.ipynb       # Jupyter notebook (EDA + training)
├── models/
│   ├── spam_svm_model.pkl    # Trained SVM model
│   └── tfidf_vectorizer_adv.pkl
├── requirements.txt          # Project dependencies
└── README.md


✔ Makes repo easy to understand  
✔ Looks professional  

---

## 5️⃣ How the System Works (Conceptual Explanation)

```markdown
## 🧠 How It Works

1. Input SMS text is cleaned and lemmatized using **NLTK**
2. Text is converted into numerical features using **TF-IDF**
3. A **Linear SVM** model classifies the message as spam or ham
4. High-risk messages are intercepted by **rule-based pattern groups**
5. Final prediction is returned via the **Streamlit UI**

## 🛡 Rule-Based Spam Detection

In addition to machine learning, the system uses grouped rule-based patterns
to catch high-risk spam that ML models may miss.

Pattern groups include:
- Financial scams (e.g. “earn $5000 per week”, “work from home”)
- Phishing attempts (e.g. “verify your bank details”, “account suspended”)
- Lottery / prize spam

Rules are applied **before** the ML model and only override predictions to spam,
making the system safer and more robust.


## ▶️ Run the App Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-detection.git
cd sms-spam-detection

install dependencies
pip install -r requirements.txt

Run the streamlit app
streamlit run streamlit_app.py

## 📊 Dataset

The model is trained on the **SMS Spam Collection Dataset**, with additional
synthetic spam samples added to improve detection of financial and phishing scams.

## 📄 License

This project is intended for educational and portfolio purposes.
