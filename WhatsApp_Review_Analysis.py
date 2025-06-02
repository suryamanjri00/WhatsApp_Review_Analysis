#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install wordcloud')


# In[3]:


# Load Library
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pprint
from wordcloud import WordCloud

plt.style.use('ggplot')

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[6]:


# Load Data
# Load data
data_dir = 'data.csv'
data = pd.read_csv(data_dir)

# Sample data
data.head(2)


# In[7]:


# Checking dataset shape
data.shape


# In[8]:


# Checking dataset columns
data.columns


# In[9]:


# Data Cleaning 🧹
# Select only the 'text' and 'score' columns
data = data[['date', 'text', 'score']]
# Remove rows with missing values
data = data.dropna()

# Confirm that duplicates have been removed
print("Number of null values after removal:\n", data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Confirm that duplicates have been removed
print("Number of duplicates after removal:", data.duplicated().sum())


# In[10]:


# Preprocess text data
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['text'] = data['text'].apply(preprocess_text)

# Display the first few rows of the cleaned dataset
print(data.head(5))


# In[11]:


# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract month and year into separate columns
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

data.head(5)


# In[12]:


# Data Visulaization 📊
# Set the style for seaborn
sns.set_style("whitegrid")

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plotting countplot (Distribution of Scores)
sns.countplot(data=data, x='score', palette='viridis', ax=axs[0])
axs[0].set_xlabel('Score')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Distribution of Scores')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plotting lineplot (Score vs Year)
sns.lineplot(data= data, x='year', y='score', marker='o', markersize=8, color='red', ax=axs[1])
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Score')
axs[1].set_title('Score vs Year Graph')
axs[1].grid(True)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[13]:


# Combine all text data into a single string
text_combined = ' '.join(data['text'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_combined)

# Plot word cloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud')
plt.axis('off')
plt.show()


# In[28]:


import nltk

# Download the punkt tokenizer data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download the punkt_tab resource as suggested by the error message
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Download the averaged_perceptron_tagger resource for POS tagging
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Download the specific English averaged_perceptron_tagger resource
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
# Download the maxent_ne_chunker_tab resource for NER chunking
try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')

# Download the 'words' resource for named entity recognition
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Download the vader_lexicon resource for VADER sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt')
except LookupError:
    nltk.download('vader_lexicon')


# In[29]:


# Download the punkt tokenizer data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download the punkt_tab resource as suggested by the error message
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Download the averaged_perceptron_tagger resource for POS tagging
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Download the specific English averaged_perceptron_tagger resource
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Download the maxent_ne_chunker_tab resource for NER chunking
try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')

# Download the 'words' resource for named entity recognition
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Text Preparation NLTK 📝
# Tokenize the 'text' column
data['tokens'] = data['text'].apply(word_tokenize)

# Add POS tags for tokens
data['pos_tags'] = data['tokens'].apply(pos_tag)

# Add chunking (Named Entity Recognition)
data['chunks'] = data['pos_tags'].apply(ne_chunk)

# Display the DataFrame with tokens
pprint.pprint(data['chunks'].head(1))


# In[30]:


# VADER sentiment scoring 🔧
# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

res = []  # List to store results

# Iterate over each row in the DataFrame using tqdm for progress tracking
for i, row in tqdm(data.iterrows(), total=len(data), desc="Sentiment Analysis"):
    text = row['text']  # Get the text from the 'Text' column

    # Run sentiment analysis on the text
    scores = sia.polarity_scores(text)

    # Store the results
    res.append(scores)


# In[31]:


# Remove columns containing date, month, and year from the original DataFrame 'data'
data = data.drop(columns=['date', 'month', 'year'])

# Convert the list of sentiment analysis results to a DataFrame and transpose it
vaders = pd.DataFrame(res)

# Reset the index of 'vaders'
vaders.reset_index(inplace=True)

# Merge 'vaders' with the original DataFrame 'data' using the index
merged_data = vaders.merge(data, left_index=True, right_index=True, how='left')

# Sample of merged_data
merged_data.head(5)


# In[32]:


# Build Insights 📈
# Plotting the mean compound sentiment score for each score category
plt.figure(figsize=(8, 6))
sns.barplot(data=merged_data, x='score', y='compound', estimator='mean')
plt.title('Compound Score for Score')
plt.xlabel('Score Category')
plt.ylabel('Mean Compound Score')
plt.show()


# In[33]:


# Set up subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# Plotting mean positive sentiment scores for each score category
sns.barplot(data=merged_data, x='score', y='pos', estimator='mean', ax=axes[0])
axes[0].set_title('Positive vs Score')
axes[0].set_ylabel('Positive Score')
axes[0].set_xlabel('Score Category')

# Plotting mean neutral sentiment scores for each score category
sns.barplot(data=merged_data, x='score', y='neu', estimator='mean', ax=axes[1])
axes[1].set_title('Neutral vs Score')
axes[1].set_ylabel('Neutral Score')
axes[1].set_xlabel('Score Category')

# Plotting mean negative sentiment scores for each score category
sns.barplot(data=merged_data, x='score', y='neg', estimator='mean', ax=axes[2])
axes[2].set_title('Negative vs Score')
axes[2].set_ylabel('Negative Score')
axes[2].set_xlabel('Score Category')

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# In[ ]:





# In[ ]:




