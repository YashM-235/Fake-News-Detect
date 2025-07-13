## 📰 Fake News Detection Using Machine Learning and Sentiment Analysis

### 🔍 Overview

Misinformation spreads faster than facts. In today's digital world, fake news can disrupt societies, influence elections, and damage reputations. This project provides an **AI-powered web application** to help users **detect fake news in real-time**, using **machine learning** and **sentiment analysis**.

🚀 Built using Python and Flask, the app predicts if a news article is real or fake and also shows its **emotional tone** (e.g., neutral, negative, positive).

---

### 🧠 What This Project Does

* 📄 **Accepts news content** as input (headline or full text).
* 🤖 **Analyzes the content** using trained ML models (like XGBoost and Random Forest).
* ✅ **Predicts** if the news is REAL or FAKE.
* 😊 **Performs sentiment analysis** to highlight emotional tone (anger, fear, positivity, etc.).
* 📊 Shows the **confidence level** of the prediction.

---

### 🎯 Key Features

* **Real-time fake news detection** via Flask-based web interface.
* **Sentiment-aware classification** using TextBlob.
* **Machine learning models** including:

  * Logistic Regression
  * Random Forest
  * Naive Bayes
  * XGBoost (best performing 🏆)
  * LightGBM
  * Neural Networks
* **High accuracy** (Up to 93.22% with XGBoost).
* **Balanced dataset** using SMOTE for fairness.
* **Interpretability** with word count, character count, sentiment score.

---

### 🧰 Technologies Used

* **Backend:** Python, Flask
* **Machine Learning Libraries:** Scikit-learn, XGBoost, LightGBM
* **Text Processing:** NLTK, TextBlob
* **Data Handling:** Pandas, NumPy
* **Visualization (optional):** Matplotlib, Seaborn

---

### 📊 Dataset Used

* **Source:** Kaggle ([Real and Fake News Dataset](https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset))
* **Size:** \~7,800 articles
* **Columns:** Title, Text, Label (REAL or FAKE)

---

### 📈 Model Performance 

| Model          | Accuracy | F1 Score | ROC-AUC |
| -------------- | -------- | -------- | ------- |
| XGBoost (Best) | 93.22%   | 92.32%   | 97.89%  |
| LightGBM       | 92.83%   | 92.90%   | 98.23%  |
| Random Forest  | 91.80%   | 91.91%   | 97.56%  |
| Naive Bayes    | 89.36%   | 89.46%   | 95.24%  |
| Neural Network | 91.10%   | 91.67%   | 97.53%  |

---

### 🤔 Why Sentiment Analysis?

Fake news tends to carry **strong emotional tones**—often negative like fear, anger, or outrage. By integrating sentiment analysis, the model not only classifies articles but also **understands their emotional intent**, making predictions more reliable.

---

### 📷 Output

<img width="866" height="466" alt="image" src="https://github.com/user-attachments/assets/06ec82b1-3ba2-478f-8f8d-8d70fbfe38d8" />

<img width="872" height="466" alt="image" src="https://github.com/user-attachments/assets/8f862bc8-9d78-4a61-b66b-b15ca31e0c3a" />


---

### 🛠 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python F_app.py

```

---

### ✨ Future Improvements

* 🌐 Multilingual detection
* ✅ Integration with real-time fact-check APIs
* 📱 Mobile-friendly UI
* 🧠 Advanced deep learning models (e.g., BERT, LSTM)

---

### 📜 Citation

If you find this project helpful in your research, feel free to cite our **IIT Roorkee SocPros 2025 Conference Paper**:

> Yash Mehta, *Fake News Detection Using Machine Learning and Sentiment Analysis Integrated in a Flask Web Application*, Proceedings of SocPros 2025, IIT Roorkee. Pg.89-90 (https://socpros2025.iitr.ac.in/wp-content/uploads/2025/02/BoA_ScropoS_2025_Final.pdf)
---

### 🤝 Contributing

Contributions, suggestions, and improvements are welcome! Feel free to fork the repo and submit a PR.

---

### 📫 Contact

**Author:** Yash Mehta
**LinkedIn:** [linkedin.com/in/yashmehta](#)
**Email:** [yash.dlw@gmail.com](#)

---
