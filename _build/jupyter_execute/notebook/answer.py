#!/usr/bin/env python
# coding: utf-8

# In[1]:


male_list=["man","men","his","his","he","male","masculine"]
female_list=["woman","women","her","hers","she","female","feminine"]


# In[2]:


import numpy as np
male_vec=[]
for i,j in zip(male_list,female_list):
    male_vec.append(model[i]-model[j])
male_vec=np.array(male_vec)
male_vec=np.mean(male_vec,axis=0)


# In[ ]:


for sport in sports:
    print(sport,get_angle(model[sport],male_vec,degree=True))


# In[ ]:





# In[ ]:


employment_pair=[("employer","employee"),("employers","employees"),("owner","worker"),
                    ("owners","workers"),("manager","worker"),("managers","staff"),
                    ("boss","worker"),("bosses","workers"),("supervisor","staff"),
                    ("supervisors","staff")]


# In[ ]:


employment_vec=create_vector(employment_pair)


# In[ ]:


cosine_similarity(employment_vec.reshape(1,-1),affluence_vec.reshape(1,-1))


# In[ ]:


ocuupation=["engineer","nurse"]
male_word =[i[0] for i in gender_pair]
female_word =[i[1] for i in gender_pair]
np.linalg.norm(model["engineer"] - model["male"]) - np.linalg.norm(model["engineer"] - model["female"])
np.linalg.norm(model["nurse"] - model["male"]) - np.linalg.norm(model["nurse"] - model["female"])


# In[ ]:


import torch
import torch.nn.functional as F

batch_size = 1
sequence_length = 3
embedding_dim = 4
seed=1234

Q = torch.rand(sequence_length, embedding_dim)
K = torch.rand(sequence_length, embedding_dim)
V = torch.rand(sequence_length, embedding_dim)

attn_output = F.scaled_dot_product_attention(Q, K, V)


# In[ ]:


import torch
import torch.nn.functional as F

sequence_length = 3
embedding_dim = 4

Q = torch.rand(sequence_length, embedding_dim)
K = torch.rand(sequence_length, embedding_dim)
V = torch.rand(sequence_length, embedding_dim)

# Step 1: Compute dot product of Q and K
dot_product = torch.matmul(Q, K.t())

# Step 2: Scale the dot product
scale_factor = torch.sqrt(torch.tensor(embedding_dim).float())
scaled_dot_product = dot_product / scale_factor

# Step 3: Apply softmax to get attention weights
attention_weights = F.softmax(scaled_dot_product, dim=-1)

# Step 4: Apply the attention weights to V
attn_output = torch.matmul(attention_weights, V)

