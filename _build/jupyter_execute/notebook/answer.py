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

