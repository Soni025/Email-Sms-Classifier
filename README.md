# 📧 Email-SMS-Classifier

An intelligent **Email & SMS Spam Detection System** built using **Machine Learning** that classifies incoming messages as **Spam** or **Not Spam**. The application leverages **Natural Language Processing (NLP)** techniques along with the **Multinomial Naive Bayes** algorithm to deliver accurate and efficient spam detection through an interactive **Streamlit** web interface.

---

## 🚀 Features

* Detects Spam and Non-Spam Email/SMS messages
* NLP-based text preprocessing pipeline
* TF-IDF Vectorization for feature extraction
* Multinomial Naive Bayes classification
* Interactive Streamlit web application
* Fast and lightweight prediction system

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn, Multinomial Naive Bayes
* **Natural Language Processing:** NLTK
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```text
Email-SMS-Classifier/
│── app.py                  # Streamlit application
│── model.pkl               # Trained spam detection model
│── vectorizer.pkl          # TF-IDF Vectorizer
│── spam.csv                # Dataset
│── requirements.txt        # Dependencies
│── sms-spam-detection.ipynb # Model training notebook
│── README.md
```

---

## ⚙️ Working

1. User enters an Email or SMS message.
2. The message undergoes text preprocessing:

   * Lowercasing
   * Tokenization
   * Stopword removal
   * Stemming
3. The cleaned text is transformed using **TF-IDF Vectorization**.
4. The trained **Multinomial Naive Bayes** model predicts whether the message is **Spam** or **Not Spam**.
5. The prediction is displayed instantly on the Streamlit interface.

---

## 📊 Machine Learning Pipeline

```text
Raw Message
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Spam / Not Spam Prediction
```

---

## 📈 Dataset

* SMS Spam Collection Dataset
* Contains labeled spam and legitimate (ham) messages
* Used for training and evaluating the spam detection model

---

## 💡 Applications

* Email Spam Filtering
* SMS Fraud Detection
* Business Communication Screening
* Customer Support Automation
* Messaging Security Systems

---

## 🔮 Future Enhancements

* Support multiple languages
* Deep Learning-based spam detection
* Email attachment analysis
* Real-time API deployment
* Confidence score for predictions
* Model retraining with new datasets

---

## ⚡ Installation

```bash
git clone https://github.com/your-username/Email-SMS-Classifier.git

cd Email-SMS-Classifier

pip install -r requirements.txt

streamlit run app.py
```

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository, open an issue, or submit a pull request.

---
