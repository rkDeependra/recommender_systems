# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:45:36 2019

@author: rk
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### functions #######

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

def combine_features(row):
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

##################################################

##Reading CSV File

df = pd.read_csv("E:/spyder/movie_recommender2/movie_dataset.csv")

#print(df.columns)
#Selecting Features
#print(df.head())

features = ['keywords','cast','genres','director']

#combines all selected features

for feature in features:
	df[feature] = df[feature].fillna('')

#print(df.head())
df["combined_features"] = df.apply(combine_features,axis=1)
#print(df.head())
#print("Combined Features:", df["combined_features"].head())

#made count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

#find the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 


########################### select a movie of users choice #################
#let
movie_user_likes = "Pirates of the Caribbean: At World's End"

#index of this movie from its title

movie_index = get_index_from_title(movie_user_likes)
print("movie user like is :",movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

#list of similar movies in descending order of similarity score

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
print("similar/recommended top 10 movies are :")
#titles of top 10 similar movies
i=0
for element in sorted_similar_movies:
		print(get_title_from_index(element[0]))
		i=i+1
		if i>10:
			break