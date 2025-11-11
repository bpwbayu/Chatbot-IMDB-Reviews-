# Chatbot-IMDB-Reviews-
A project to classify IMDB movie reviews using machine learning and a chatbot interface.

🤖 Chatbot IMDB Reviews 
The aim is to develop chatbots and machine learning models that can classify film review sentiments from the IMDB dataset and provide appropriate automated responses based on the results of the sentiment analysis.

📁Dataset

Source: Kaggle - IMDB Dataset of 50K Movie Reviews
The dataset offers a balanced set of labelled text samples, making it ideal for training and evaluating sentiment analysis models.

⚙ Tools 

Python (Pandas, NumPy, Scikit-Learn, Re, Joblib,  Matplotlib, Seaborn, NLTK, Streamlit )
Jupyter Notebook/VS Code 

🎯 Purpose of Analysis

Provide prompt and relevant responses to customer inquiries or complaints.
Identify positive, neutral, and negative sentiments to gauge customer satisfaction and detect potential issues.
Support data-driven insights for content producers, marketers, and researchers interested in audience sentiment trends.

🔎 Steps 

1. Data Loading & Exploratory Data Analysis (EDA)
Reading datasets using Pandas
Checking missing values and descriptive statistics
Examine message length, inbound distribution, number of reviews per date
2. Feature Engineering
Text cleaning and lemmatization
Create additional features for sentiment scores then convert scores into sentiment labels
One-hot encoding for additional features
3. Data Splitting & Scaling
Split the data into train (80%) and test (20%)
The data was split using stratification to maintain balanced proportions of each sentiment label, and the text was then transformed into numerical representations using TF-IDF before being fed into the models.
4. Modeling
Model used:
Multinomial Naive Bayes
Logistic Regression
Random Forest Classifier
5. Evaluation
Classification Report (Precision, Recall, F1-support)
Confusion Matrix
Accuracy Score
📊 Visualization
The sentiment distribution shows a balanced dataset between positive and negative reviews, each with roughly equal frequency.
The text length distribution indicates that most reviews are short to medium-length, with a few very long reviews extending beyond 10,000 characters.
🧠Insight
The project successfully developed a chatbot integrated with a machine learning model capable of classifying IMDb movie review sentiments accurately. 
Through machine learning, the Logistic Regression model achieved the highest and most consistent performance (accuracy = 0.8975), outperforming Naive Bayes and Random Forest.
🚀 How to Run
Ensure all required packages are installed:
pip install pandas numpy scikit-learn re joblib matplotlib seaborn nltk streamlit 
Run the notebook in Jupyter or VS Code:
jupyter notebook Project_ChatbotCustomerSupport.ipynb
Run the Streamlit:
streamlit run app.py
