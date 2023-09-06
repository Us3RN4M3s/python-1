#!/usr/bin/env python
# coding: utf-8

# In[62]:


job = ['stringless', "useful"]


# In[63]:


cloud = ['funny', 'weak', 'data', 'words']
item = cloud [3] #gets 4th item in list 
print(item)


# In[64]:


for items in cloud:
    print(items + ' its strong' + {job}) #error because {job} is an unhashable type of "list" 


# In[65]:


kid = ['wow']
print(job + kid) # prints stringless, useful, and wow because the lists job and kid are concatenated 
cloud.append("hahaha games") #appends this string to the cloud list 
print(item + (" what")) #concatenates this string to the list "item" and prints 
print (cloud, item) #prints both lists 


# In[66]:


def funky(music = 'disco'): #a music function default is disco
    print(f"{music} starts playing") #takes parameter 


# In[67]:


funky("rock") #rock parameter given 


# In[68]:


funky() #default parameter given 


# In[10]:


ids = {"john": "pack", "sarah" :"stick", "laura" :"jump", "katy": "act"} #dictionary of ids
print(ids)


# In[89]:


x = 7
if (x) >= 1: #if x is equal or greater than 1, add 1 to x
    x = x+1
    print(x)
else:        #if x is anything else, multiply x by 9
    x = x*9
    print(x)


# In[12]:


addict = {"judge": "jobless", "soda": "sprite", "money": 134.23} #dictionary 


# In[13]:


for first in addict.keys(): #loops through the dictionary and concatenates "is funny" to each key printed
    print(f"{first} is funny")


# In[14]:


for second in addict.values(): #loops through dictionary and concatenates "is not funny" to each value printed
    print(f"{second} is not funny")


# In[ ]:





# In[90]:


i = 1
while i < 600:
  print(i)
  if i == 500:
    break
  i += 1


# In[76]:


import random


def simplefunc():
    print("this is a simple function")
def complexfunc():
    print(random.randint(0,255),  "this is a big number" , addict.values()) 


# In[77]:


simplefunc()


# In[78]:


complexfunc()


# In[79]:


class hero():
    def __init__(self,name,atk,hp,defense):
        self.name = name
        self.atk = atk
        self.defense = defense
        self.hp = hp
        print(f"your hero is {self.name}, with {self.atk} attack, {self.defense} defense, and {self.hp} health")


hero("jack", "50","89", "100")



# In[51]:


class Car:
    def __init__(self, make, model, year,enginesize,fueleco):
        self.make = make
        self.model = model
        self.year = year
        self.enginesize = enginesize
        self.fueleco = fueleco
        
        self.fuel_capacity = 18
        self.fuel_level = 0
        
    def fill_tank(self):
        self.fuel_level = self.fuel_capacity
        print("gas tank all filled up")
    def drive(self):
        print("the car is put into drive and moves")


# In[80]:


my_car = Car("toyota","camry","2022","4cylinder","35mpg")
print(my_car.fueleco)
print(my_car.year)
my_car.drive()
my_car.year = "2018"
print(my_car.year)


# In[81]:


filename = 'funny.txt'
with open(filename) as file_object:
    lines = file_object.readlines()
for line in lines:
    print(line)


# In[96]:


filename = 'funny1.txt' 
with open(filename, 'w') as file_object:
    file_object.write("\n This is the fittnessgram pacer test script.") #changes the text in the funny.txt file


# In[97]:


prompt = "whats up?" #asks user "whats up" but when giving a normal reponse the program cant hear, but giving an int, the program says good
response = input(prompt)

try:
    response = int(response)
except ValueError:
    print("what was that?")
else:
    print("thats good to hear")


# In[59]:


import matplotlib.pyplot as plt
x_values = [0,1,2,3,4,5,6,10,1,4,7,9,3,4,5,6,9]
squares = [0,1,4,9,16,25,36,100,8,4,7,8,56,78,23,78,62]
fix, ax = plt.subplots()
ax.plot(x_values,squares)
plt.show


# In[98]:


import matplotlib.pyplot as plt
x_values = list(range(500)) #creates a list called x_values with a index range of 1 - 500
numbers = [x**10+(5*4) for x in x_values] # numbers is the list and gets x^10 + (20) and loops through the list 
plt.style.use('dark_background') #a style for the plot 
fig, ax = plt.subplots()
ax.scatter(x_values,numbers, s=10) #use a scatter plot with x_values in x, numbers in y, the s parameter changes the size of the dots 
plt.show() #displays plot 


# In[ ]:







# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




