import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
#         'Age':[27, 24, 22, 32],
#         'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
#         'Qualification':['Msc', 'MA', 'MCA', 'Phd']}


# df = pd.DataFrame(data)

# print(df["Name"][0])

# nba_data = pd.read_csv('nba.csv', index_col='Name')

# first = nba_data.loc["Carmelo Anthony"]

# print(first)

# # making data frame from csv file
# data = pd.read_csv("nba.csv", index_col ="Name")

# # retrieving rows by iloc method 
# row2 = data.iloc[3] 

# print(row2)

# dictionary of lists
dict = {'name':["aparna", "pankaj", "sudhir", "Geeku"],
        'degree': ["MBA", "BCA", "M.Tech", "MBA"],
        'score':[90, 40, 80, 98]}
  
# creating a dataframe from a dictionary 
df = pd.DataFrame(dict)

# #creates an list of the columns for easy access
# columns = list(df)

# print(columns)
 
# for i in columns:
 
#     # printing the third element of the column
#     print (df[i][2])

# df_array = df.to_numpy()

# print(len(df_array[0]))

# for value in df_array[0]:
#     print(value)

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression().fit(x, y)

r_sq = model.score(x,y)

print(f"The score of this linear regresion is: {r_sq}")