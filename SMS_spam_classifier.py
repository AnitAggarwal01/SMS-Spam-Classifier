
# coding: utf-8

# In[16]:


import sys
import nltk
import sklearn
import pandas
import numpy

print('Python : {}'.format(sys.version))
print('NLTK : {}'.format(nltk.__version__))
print('Scikit-Learn : {}'.format(sklearn.__version__))
print('Pandas : {}'.format(pandas.__version__))
print('NumPy : {}'.format(numpy.__version__))


# ## 1.Importing the Dataset

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_table('SMSSpamCollection',header = None, encoding='utf-8')


# In[3]:


classes = df[0]
print(classes.value_counts())
print(df)


# ## 2. Preprocessing the Dataset

# ### Label Encoding

# In[7]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y=encoder.fit_transform(classes)
print(Y[0:10])


# ### Replacing email,web addresses, money symbols, phone numbers with Text 

# In[8]:


text_messages = df[1]
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
processed = processed.str.replace(r'£|\$', 'moneysymb')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')


# ### Removing punctuations and whitespaces

# In[10]:


processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')


# ### Converting to lowercase

# In[11]:


processed = processed.str.lower()


# ### Removing Stopwords

# In[12]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
processed = processed.apply(lambda x:' '.join(term for term in x.split() if not term in stop_words))


# ### Lemmatization

# In[13]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
processed = processed.apply(lambda x:' '.join(lemmatizer.lemmatize(term) for term in x.split()))


# In[14]:


print(processed)


# ## 3. Feature Engineering

# ### Word Tokenization

# In[17]:


from nltk.tokenize import word_tokenize
all_words = []
for message in 
processed:
    words = word_tokenize(message)
    for word in words:
        all_words.append(word)
all_words = nltk.FreqDist(all_words)


# In[18]:


word_features = list(all_words.keys())


# ### Function for Finding Features in a Message

# In[19]:


def find_features(message):
    message = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = word in message
    return features


# In[21]:


features =  find_features(processed[0])
for key,value in features.items():
    if value == True:
        print(key)


# ### Generating Feature Vectors for all messages

# In[104]:


messages = list(zip(processed,Y))

seed = 1
np.random.seed = seed
np.random.shuffle(messages)

featuresets = [(find_features(text),label) for (text, label) in messages]


# ### Splitting the dataset into training/testing data

# In[105]:


from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(featuresets,test_size = 0.25, random_state = seed)


# ## 4. Scikit Learn Classifiers

# In[108]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

classifiers = [KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression(),SGDClassifier(),MultinomialNB(),SVC()]
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGDC Classifier","Naive Bayes", "SVM Linear"]
models = list(zip(names,classifiers))


# In[111]:


from nltk.classify.scikitlearn import SklearnClassifier

for name, classifier in models:
    nltk_model = SklearnClassifier(classifier)
    nltk_model.train(training_set)
    accuracy = nltk.classify.accuracy(nltk_model,test_set)*100
    print('{} Accuracy: {}'.format(name,accuracy))


# ### Results : Best Accuracy is with Logistic Regression Model

# Trying Ensembling

# In[118]:


from sklearn.ensemble import VotingClassifier
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models[1:5], voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training_set)
Accuracy = nltk.classify.accuracy(nltk_ensemble, test_set)*100
print("Voting Classifier: Accuracy: {}".format(Accuracy))


# ### Result: Failed Ensembling decreases accuracy

# In[126]:


txt_features, labels = zip(*test_set)
nltk_model = SklearnClassifier(LogisticRegression())
nltk_model.train(training_set)
prediction = nltk_model.classify_many(txt_features)
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['not Spam', 'Spam']],
    columns = [['predicted', 'predicted'], ['not Spam', 'Spam']])


# In[127]:


prediction = nltk_ensemble.classify_many(txt_features)
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['not Spam', 'spam']],
    columns = [['predicted', 'predicted'], ['not Spam', 'spam']])

