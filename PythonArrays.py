#!/usr/bin/env python
# coding: utf-8



# # NumPy

# In[1]:


import numpy as np


# ### Simple arrays and nested arrays

# In[36]:


a = np.array([1,2,3,4,5,6]) #an array
b = np.array([9.67,7,2.4],dtype = float) #float type array
c = np.array([[(1.5,2,4),(7,3,5)],[(3,2,1),(4,5,6)]],dtype =float) #float type nested arrays 
print(a)
print(b)
print(c)


# ### Array of zeros and ones 

# In[65]:


g = np.zeros((3,4,2)) #array of zeroes 
g


# In[68]:


y = np.ones((7,2,5),dtype=np.float32) #array of ones 
y


# ### Arrays made evenly spaced between elements 

# In[42]:


h = np.arange(0,1000,32.156) #array of evenly spaced values (going by steps)
print(h)


# In[159]:


abc = np.linspace(0,1000,20) #array evenly spaced values, (x,y,z) where x is starting,y is end, z is total samples
abc


# ### saving and loading

# In[86]:


np.savez('array.npz', abc) #saving array to disk


# In[87]:


np.load('array.npz') #loading array from disc and displaying array
abc


# In[90]:


np.savetxt("myarray.txt",abc,delimiter=',') #saving array abc as txt using , as delimiter


# In[91]:


np.loadtxt('myarray.txt') #loading array abc from txt 


# # Pandas practice 

# In[94]:


import pandas as pd


# ### Series 

# In[95]:


s = pd.Series([3,-5.6,2,-1.1], index = ['sea','ocean','soda','pizza']) # a series with numbers and items


# In[96]:


s


# ### dataframe with different users, access level and time spent and favorite music player

# In[251]:


data = {'user': ['user1','user2','user3'], 
        'access': ['admin', 'user','admin'],
        'time/week': ['40','13','34'],
       'favorite_musicplayer':['Spotify','iTunes','VLC']}
frame = pd.DataFrame(data,columns=["user",'access','time/week','favorite_musicplayer'])
frame


# In[121]:


frame.drop('access',axis=1) #drops the column 'access'


# In[127]:


frame.drop('time/week',axis=1) # drops the column 'time_spent_perweek'


# In[248]:


frame.iloc[[0],[0]] #gets single value by row and column


# In[249]:


frame.loc[[0],['favorite_musicplayer']] # gets single value by column label


# In[250]:


frame.loc[[2],['access']] #gets value from index 2 with column access


# In[135]:


frame.drop([0,1]) # drops row with index 0 and 1


# In[162]:


t = pd.Series([y,abc,h]) # series with index 0,1,2 with arrays y, abc, and h in each respective index


# In[163]:


y


# In[164]:


abc


# In[165]:


h


# In[143]:


t


# In[167]:


tframe = pd.DataFrame(data,columns=[abc]) #creates dataframe using the array abc made earlier 
tframe


# In[175]:


dframe = pd.DataFrame(data,columns=[h])


# In[176]:


dframe


# In[177]:


rframe = pd.DataFrame(data,columns=[y]) #error because y is a multidimensional array and cant be displayed


# # Reshaping arrays, sorting, indexing,concatenating 

# In[206]:


y.shape


# In[210]:


xyz = y.reshape(70)
xyz.shape


# In[211]:


rframe = pd.DataFrame(data,columns=[xyz])
rframe


# In[235]:


revabc = abc[::-1] #reverse order of array


# In[236]:


revabc


# In[242]:


revabc[revabc<120] #displays all elements under 120


# In[225]:


h[24] #gets 24th element in array h


# In[227]:


h[7:14] #gets index 7 up to index 14 in array h


# In[228]:


h[h>300] #displays all elements above 300


# In[246]:


guy = np.concatenate((h,revabc,xyz),axis=0) #concatenates arrays h,revabc, and xyz
guy.sort() #sorts the elements in this new array
guy


# In[ ]:




