# 🕵️ Fake Job Postings Detection

This project detects fraudulent job postings using **Natural Language Processing (NLP)** and **Machine Learning**.  
The dataset used is `fake_job_postings.csv` (from Kaggle).

---

## 📌 Project Workflow

1. **Data Preprocessing**
   - Removed missing values in `description`
   - Cleaned text:
     - Removed HTML tags
     - Converted to lowercase
     - Removed punctuation, numbers, and stopwords
     - Lemmatized words

2. **Train-Test Split**
   - Data was split into training (80%) and testing (20%) sets
   - Used both raw `description` and cleaned `final_text` for experiments

3. **Feature Extraction**
   - Used **TF-IDF Vectorization** (top 5000 features)

4. **Model**
   - Logistic Regression (baseline)
   - Predictions compared with actual labels

---




