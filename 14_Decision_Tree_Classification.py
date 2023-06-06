#!/usr/bin/env python
# coding: utf-8

# # Classification Task dengan Decision Tree
# 
# Referensi: [https://en.wikipedia.org/wiki/Decision_tree_learning](https://en.wikipedia.org/wiki/Decision_tree_learning)

# ## Konsep Dasar
# 
# ### Terminology: root node, internal node, leaf node
# 
# <div>
# <img src="./images/decision_tree_example.png" width="600">
# </div>

# a### Gini Impurity
# 
# <div>
#     <img src="./images/gini_example.png" width="400">
# </div>
# 
# #### Ruas Kiri:
# 
# $
# \begin{align*} 
# G &= 1 - \sum_i^n P_i^2 \\
#   &= 1 - P(biru)^2 \\
#   &= 1 - (\frac{4}{4})^2 = 0
# \end{align*}
# $
# 
# #### Ruas Kanan:
# 
# $
# \begin{align*}
# G &= 1 - \sum_i^n P_i^2 \\
#   &= 1 - (P(biru)^2 + P(hijau)^2)\\
#   &= 1 - ( (\frac{1}{6})^2 + (\frac{5}{6})^2 ) = 0.278
# \end{align*}
# $
# 
# 
# #### Average Gini Impurity:
# 
# $
# \begin{align*}
# G &= \frac{4}{4+6} \times 0 + \frac{6}{4+6} \times  0.278 \\
#   &= 0.1668
# \end{align*}
# $

# ### Information Gain
# 
# <div>
#     <img src="./images/information_gain.png" width="500">
# </div>

# ### Membangun Decision Tree
# 
# <div>
#     <img src="./images/build_decision_tree.png" width="900">
# </div>
# 
# $
# \begin{align*} 
# G &= 1 - (P(apple)^2 + P(grape)^2 + P(lemon)^2) \\
#    &=1 - ( (\frac{2}{5})^2 + (\frac{2}{5})^2 + (\frac{1}{5})^2 ) \\
#    &= 0.63
# \end{align*}    
# $

# In[2]:


1 - ( (2/5)**2 + (2/5)**2 + (1/5)**2 )


# Persiapan dataset

# In[3]:


import pandas as pd

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

# Column labels.
# These are used only to print the tree.
header = ["Color", "Diameter", "Label"]

pd.DataFrame(training_data, columns=header)


# In[4]:


from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

print(f'Dimensi Feature: {X.shape}')
print(f'Class: {set(y)}')


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)


# ## Classification dengan `DecisionTreeClassifier`

# In[6]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4)

model.fit(X_train, y_train)


# ## Visualisasi Model

# In[7]:


import matplotlib.pyplot as plt
from sklearn import tree

plt.rcParams['figure.dpi'] = 85
plt.subplots(figsize=(10, 10))
tree.plot_tree(model, fontsize=10)
plt.show()


# ## Evaluasi Model

# In[8]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[ ]:




