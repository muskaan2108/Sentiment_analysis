#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import re
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()


# In[2]:


sw = stopwords.words('english')


# In[3]:


# read the positvie text data
pos_rev = pd.read_csv('pos.txt' , sep='\n' , header = None , encoding = 'latin-1')

# adding a tareget column
pos_rev['mood'] = 1.0
pos_rev = pos_rev.rename(columns = {0:'review'})
pos_rev


# In[4]:


# read the negative text data
neg_rev = pd.read_csv('negative.txt' , sep='\n' , header = None , encoding = 'latin-1')

# adding a target column
neg_rev['mood'] = 0.0
neg_rev = neg_rev.rename(columns = {0:'review'})
neg_rev


# In[5]:


pos_rev['review'] = pos_rev['review'].apply(lambda x : x.lower())
pos_rev['review'] = pos_rev['review'].apply(lambda x : re.sub(r"@\S+" , "" , x))
pos_rev['review'] = pos_rev['review'].apply(lambda x : lemma.lemmatize(x,pos='v'))
pos_rev['review'] = pos_rev['review'].apply(lambda x : x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
pos_rev['review'] = pos_rev['review'].apply(lambda x : " ".join([word for word in x.split() if word not in (sw)]))
pos_rev


# In[6]:


neg_rev['review'] = neg_rev['review'].apply(lambda x : x.lower())
neg_rev['review'] = neg_rev['review'].apply(lambda x : re.sub(r"@\S+" , "" , x))
neg_rev['review'] = neg_rev['review'].apply(lambda x : lemma.lemmatize(x,pos='v'))
neg_rev['review'] = neg_rev['review'].apply(lambda x : x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
neg_rev['review'] = neg_rev['review'].apply(lambda x : " ".join([word for word in x.split() if word not in (sw)]))


# In[7]:


com_rev = pd.concat([pos_rev , neg_rev] , axis = 0).reset_index()
com_rev


# In[8]:


X_train , X_test, y_train, y_test = train_test_split(com_rev['review'].values , com_rev['mood'].values , test_size = 0.2 , random_state = 101)


# In[9]:


train_data = pd.DataFrame({'review':X_train , 'mood':y_train})
test_data = pd.DataFrame({'review':X_test , 'mood':y_test})


# In[10]:


train_data


# In[11]:


test_data


# In[12]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data['review'])
test_vectors = vectorizer.transform(test_data['review'])


# In[13]:


from sklearn import svm
from sklearn.metrics import classification_report


# In[14]:


classifier = svm.SVC(kernel='linear')
classifier.fit(train_vectors , train_data['mood'])


# In[15]:


pred = classifier.predict(test_vectors)


# In[16]:


report = classification_report(test_data['mood'] , pred , output_dict=True)
print(f"positive {report['1.0']['recall']}")
print(f"negative {report['0.0']['recall']}")


# In[17]:





# In[18]:


import joblib
model_file_name = 'netflix_svm_model.pkl'
vectorizer_filename = 'netflix_vector.pkl'
joblib.dump(classifier , model_file_name)
joblib.dump(vectorizer , vectorizer_filename)


# In[19]:


# loading a model
vect = joblib.load('netflix_vector.pkl')
clf = joblib.load('netflix_svm_model.pkl')


# In[20]:


a = input('enter the review : ')
b = vect.transform([a])
predicted=clf.predict(b)
if predicted[0]==0:
    print('negative')
else:
    print('positive')    


# In[ ]:





# In[ ]:




