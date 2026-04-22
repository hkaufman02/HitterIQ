# ⚾ HitterIQ

AI-powered MLB hitter analytics dashboard built using machine learning.

---

## 🚀 Features

- 🔥 **Breakout Probability Prediction**
- 📈 **OPS Projection Model**
- 🧠 **AI Score ranking system**
- 🧬 **Player similarity engine**
- 📊 **Interactive Streamlit dashboard**
- 🎯 **Age-based filtering (young, prime, veterans)**
- 📉 **Model insights (feature importance)**

---

## 🧠 How It Works

HitterIQ uses engineered baseball statistics to evaluate hitter performance and future potential.

Key inputs include:
- OPS, wOBA  
- Power, discipline, and production metrics  
- Historical performance trends and rolling averages  

These features feed into machine learning models that:
- predict breakout hitters  
- project future offensive performance (OPS)  
- rank players using a custom AI scoring system  

---

## 🛠️ Tech Stack

- Python  
- pandas / numpy  
- scikit-learn  
- Streamlit  
- Plotly  

---

## 📊 Example Use Cases

- Identify breakout MLB hitters  
- Compare player performance profiles  
- Analyze offensive trends over time  
- Explore which features drive predictions

---

## 💡 Future Improvements

- **Integrate real-time MLB data** for live predictions  
- **Add player vs player comparison feature**  
- **Deploy dashboard for public access**  
- **Improve model performance** with advanced algorithms (XGBoost, tuning)  

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python src/train_classifier.py
python src/train_regressor.py
python -m streamlit run app/streamlit_app.py
