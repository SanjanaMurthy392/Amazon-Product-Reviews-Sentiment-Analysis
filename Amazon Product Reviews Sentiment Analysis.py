#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()

data = pd.read_csv("C:/Users/Lenovo/OneDrive/Documents/Reviews.csv")
print(data.head())


# In[2]:


print(data.describe())


# In[3]:


data = data.dropna()


# In[4]:


ratings = data["Score"].value_counts()
numbers = ratings.index
quantity = ratings.values

custom_colors = ["skyblue", "yellowgreen", 'tomato', "blue", "red"]
plt.figure(figsize=(10, 8))
plt.pie(quantity, labels=numbers, colors=custom_colors)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Distribution of Amazon Product Ratings", fontsize=20)
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm

nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()

data = pd.read_csv("C:/Users/Lenovo/OneDrive/Documents/Reviews.csv")

print(data.head())
print(data.columns)

tqdm.pandas()

data["Positive"] = data["Text"].progress_apply(lambda x: sentiments.polarity_scores(x)["pos"])
data["Negative"] = data["Text"].progress_apply(lambda x: sentiments.polarity_scores(x)["neg"])
data["Neutral"] = data["Text"].progress_apply(lambda x: sentiments.polarity_scores(x)["neu"])

print(data.head())


# In[6]:


x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive 😊 ")
    elif (b>a) and (b>c):
        print("Negative 😠 ")
    else:
        print("Neutral 🙂 ")
sentiment_score(x, y, z)


# In[7]:


print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)


# In[ ]:




