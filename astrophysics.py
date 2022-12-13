#!/usr/bin/env python
# coding: utf-8

# Astronomy has a rich history of data collection and recording. Data about and from celestial bodies are collected using various telescopes, photon detectors, and particle detectors. Although the entire electromagnetic spectrum is important, most observational data come from the visible/infrared (wavelengths from 400 nm to 1 mm) and radio (wavelengths from 1 cm to 1 km) portions of the spectrum.
# 
# Co-authorship network is a group of researchers connected in pairs to represent relationships. Two researchers are considered related when they have published papers in journals and conferences and published and edited books together. In such networks, researchers are called "nodes" or "nodes", and connections are called "edges". Co-authorship networks are an important class of social networks. Analysis of these networks reveals characteristics of academic communities that contribute to understanding collaboration and identifying prominent researchers.
# 
# The focus of this notebook is to analysis of structural properties in co-authored networks in astrophysics using centrality measures. 

# In[47]:


import pandas as pd 
import numpy as np 
from datetime import datetime
import sys
import ast

import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

import dask.bag as db

import networkx as nx
from networkx.algorithms.components.connected import connected_components

import json
import dask.bag as db
import os

sys.path.append("..")

from pathlib import Path

import json

from itertools import combinations
from collections import Counter
from itertools import chain
import random

from tqdm.notebook import tqdm, trange
import time    # to be used in loop iterations

import multiprocessing
import smart_open

from gensim.models.word2vec import Word2Vec

import pyvis

from pyvis.network import Network

from IPython.core.display import display, HTML

import streamlit as st


# In[48]:


ai_category_list=['astro-ph']
records=db.read_text("./arxiv-metadata-oai-snapshot.json").map(lambda x:json.loads(x))
ai_docs = (records.filter(lambda x:any(ele in x['categories'] for ele in ai_category_list)==True))
get_metadata = lambda x: {'id': x['id'],
                  'title': x['title'],
                  'category':x['categories'],
                  'abstract':x['abstract'],
                 'version':x['versions'][-1]['created'],
                         'doi':x["doi"],
                         'authors_parsed':x['authors_parsed']}

data=ai_docs.map(get_metadata).to_dataframe().compute()

data.to_excel("Astro_ArXiv_Papers.xlsx",index=False,encoding="utf-8")


# In[49]:


# Concatenate the author first and last names
data['num_authors']=data['authors_parsed'].apply(lambda x:len(x))

data['authors']=data['authors_parsed'].apply(lambda authors:[(" ".join(author)).strip() for author in authors])
data.head()


# In[50]:


data['DateTime']=pd.to_datetime(data['version'])
data['Day'] = data['DateTime'].dt.day
data['Month'] = data['DateTime'].dt.month
data['Year'] = data['DateTime'].dt.year


# In[51]:


# Filter data (2020 to 2021)

data = data[data['Year'].between(2020, 2021)]


# In[52]:


# This project focuses on individual author structure and development; whereas LIGO or VIRGO would not be taken into account.
data = data[data['num_authors'] <= 6]


# In[53]:


data['author_pairs']=data['authors'].apply(lambda x:list(combinations(x, 2)))


# In[54]:


def flattenList(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


# In[55]:


astro_authors=pd.DataFrame(flattenList(data['authors'].tolist())).rename(columns={0:'authors'})
papers_by_authors=astro_authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False)
papers_by_authors.shape


# In[56]:


papers_by_authors['Number of Papers Published'].describe()


# In[57]:


# Keeping Authors who have published more than 2 Papers
nodes_to_keep=papers_by_authors.loc[papers_by_authors['Number of Papers Published']>2,'authors'].tolist()
len(nodes_to_keep)


# ## Generating the Edges of the Co-Author Network

# In[58]:


authors_pairs=data['author_pairs'].tolist()
authors_edge_list=[item for sublist in authors_pairs for item in sublist]
authors_weighted_edge_list=list(Counter(authors_edge_list).items())
authors_weighted_edge_list=[(row[0][0],row[0][1],row[1]) for idx,row in enumerate(authors_weighted_edge_list)]
authors_weighted_edge_list[0:10]


# In[59]:


G1=nx.Graph()
G1.add_weighted_edges_from(authors_weighted_edge_list)
print(len(G1.nodes()))


# In[60]:


sub_g=nx.subgraph(G1,nodes_to_keep)
G=nx.Graph(sub_g)
print(len(G.nodes()))
isolated_node=nx.isolates(G)
len(list(isolated_node))


# In[61]:


G.remove_nodes_from(list(nx.isolates(G)))
len(G.nodes)


# In[62]:


del G1, sub_g


# In[63]:


print("Number of Nodes in Author Graph ",len(G.nodes()))
print("Number of Edges in AUthor Graph ",len(G.edges()))


# In[64]:


def getRandomWalk(graph,node,length_of_random_walk):
    start_node=node
    current_node=start_node
    random_walk=[node]
    for i in range(0,length_of_random_walk):
        current_node_neighbours=list(graph.neighbors(current_node))
        chosen_node=random.choice(current_node_neighbours)
        current_node=chosen_node
        random_walk.append(current_node)
    return random_walk


# #### For every Node in the Graph, get randomwalks . 
# #### For each node, get 10 random walks 

# In[65]:


num_sampling=10
random_walks=[]
length_of_random_walk=10
for node in tqdm(G.nodes(),desc="Iterating Nodes"):
    
    for i in range(0,num_sampling):
        random_walks.append(getRandomWalk(G,node,length_of_random_walk))


# In[66]:


deepwalk_model=Word2Vec(sentences=random_walks,window=5,sg=1,negative=5,vector_size=128,epochs=20,compute_loss=True)


# In[67]:


deepwalk_model.save("deepwalk1.model")


# In[68]:


def getSimilarNodes(model,node):
    similarity=model.wv.most_similar(node)
    similar_nodes=pd.DataFrame()
    similar_nodes['Similar_Node']=[row[0] for i,row in enumerate(similarity)]
    similar_nodes['Similarity_Score']=[row[1] for i,row in enumerate(similarity)]
    similar_nodes['Source_Node']=node
    return similar_nodes


# In[69]:


ai_authors=pd.DataFrame(flattenList(data['authors'].tolist())).rename(columns={0:'authors'})
papers_by_authors=ai_authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False)
papers_by_authors


# In[70]:


def getCoAuthorshipNetwork(graph,initial_nodes):
    total_neighbours=0
    nodes_set=[initial_nodes]
    for node in initial_nodes:
        #print(node)
        neighbours=list(graph.neighbors(node))
        total_neighbours=total_neighbours+len(neighbours)
        
        nodes_set.append(neighbours)
    print(total_neighbours)
    nodes_set=flattenList(nodes_set)
    return list(set(nodes_set))


# In[71]:


coauthor_nodes=getCoAuthorshipNetwork(G,papers_by_authors['authors'].tolist()[4:10])
print("Number of CoAuthor Nodes ",len(coauthor_nodes))


# In[72]:


coauthor_subgraph=nx.subgraph(G,coauthor_nodes)
print("number of edges in the CoAuthor Subgraph ",len(coauthor_subgraph.edges()))


# In[73]:


nx.write_gexf(coauthor_subgraph, "CoAuthor_Subgraph_Author4to10.gexf")


# In[74]:


print("number of edges in the CoAuthor Subgraph ",len(coauthor_subgraph.edges()))


# In[75]:


pyvis_nt=Network(notebook=True,height='800px', width='100%',heading='')

print("Creating PyVis from NetworkX")
pyvis_nt.from_nx(coauthor_subgraph)

print("Saving PyVis Graph")
pyvis_nt.show("Author4to10_CoAuthorGraph.html")


# In[76]:


# In[80]:


coauthor_subgraph=nx.subgraph(G,coauthor_nodes)

# fig, ax = plt.subplots()
# pos = nx.kamada_kawai_layout(coauthor_subgraph)
# nx.draw(coauthor_subgraph,pos, with_labels=True)
# st.pyplot(fig)
# st.balloons()
dot = nx.nx_pydot.to_pydot(coauthor_subgraph)
st.graphviz_chart(dot.to_string())


# In[ ]:




