# Fake-News-Prediction-
Project Name:- Fake-News-Prediction 2)Work Flow :- a) Taking News DataSet from Kaggle. b) Data Preproccessing c)Train_Test_Split d)Logistic Model e)Train Logistic Model
f)Feed New Data To a Model.
About the data set. a) id: Unique Id for a news article. b) title: Title of a news article. c) author: Author of a news article d) text: text inside article e) label: Is a numerical column where 1 = Fake News and 2 = Real News.
Importing Dependencies:- import numpy as np/import pandas as pd/import re/from nltk.corpus import stopwords/ from nltk.stem.porter import PorterStemmer/fromsklearn.feature_extraction.text import TfidfVectorizer from sklearn.model_selection import train_test_split/from sklearn.linear_model import LogisticRegression/ from sklearn.metrics import accuracy_score.
In these Data set we used LogisticsRegression because the output variable is binary form that is news can be real or fake.
Here we import nltk and download stopwords from nltk. Stopwords are a words that is not important. Further we have to remove them.
Data Preproccessing :- In Data Preproccessing we read the data with the help of Pandas , check the heading and top 5 rows of data set, check the null values, replace the null cell with empty string.
Merge the features: We will merge the two column 'title' and 'author' and make new column 'content'.
Stemming:- Stemming is the proccess of reducing a word to it's root word. Example : actor,actress,acting all three will conver into act.
Make a User Defined Function on line number 13 : line no. 1 def stemming(content): line no. 2 stemmed_content = re.sub('[^a-zA-Z]',' ',content) line no. 3 stemmed_content = stemmed_content.lower() line no. 4 stemmed_content = stemmed_content.split() line no. 5 stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] line no. 6 stemmed_content = ' '.join(stemmed_content) line no. 7 return stemmed_content                       In Line no.1 we use def to make our own function, name stemming. In Line no.2 we use 're' we import re library, re stand for Regular Expression Library it is useful for specifies a set of strings that matches it; the functions in this module let you check if a particular string matches a given regular expression. We used 'sub' function, sub basically means it substitute a certain value. '^' means exclusion , [^a-zA-Z] means take a-z and A-Z only alphabet and remove numbers and punctuation mark and put instead of them for 'content' column. In Line no.3 Convert all the word in lower case. In Line no.4 Split all the words and convert them into a list. In Line no.5 We will stemmed all the word to there root word and then apply 'for' loop to word in stemmed_content and remove stopwords that are insignificant word and choose only the word that are not stopwords. In Line no.6 Join all the word.
We apply above function on 'content' column.
Seperate data and label in X and Y.
Converting textual data to numerical data with the help of vectorizer. code: vectorizer = TfidfVectorizer() vectorizer.fit(X) X = vectorizer.transform(X) In above code Tf stand for term frequency and idf stand for invrse document frequency. Tf basically count number of particular words which is repeating in text and idf check the word which is repeating continously but it is not important.
Splitting data into train and test.
Training a model.
Fit a model on train set.
Evaluate the model with accuracy test.
Check accuracy on train and test data.
Make Prediction System with if_else condition.
