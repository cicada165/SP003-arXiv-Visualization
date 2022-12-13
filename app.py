import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 
import pyvis
from pyvis.network import Network
from itertools import combinations
from collections import Counter
import networkx as nx
from tqdm.notebook import tqdm, trange
from gensim.models.word2vec import Word2Vec
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import json
import dask.bag as db
from PIL import Image



# read csv
data = pd.read_csv("final_dataset.csv")
print(data.shape)
data = data.loc[data['Country']!="Country not found through Wiki Search"]
data['DateTime']=pd.to_datetime(data['version'])
data['Day'] = data['DateTime'].dt.day
data['Month'] = data['DateTime'].dt.month
data['Year'] = data['DateTime'].dt.year
# data['authors_parsed'] = data.authors_parsed.apply(lambda x: x[1:-1].split(','))

# for index, row in data.iterrows():
#     tmp = [word.strip("''") for word in row['authors_parsed']]
#     row['authors_parsed'] = tmp


st.set_page_config(
    page_title = 'arXiv Dashboard',
    page_icon = '✅',
    layout = 'wide'
)

# dashboard title

st.title("arXiv Dashboard")

# creating a single-element container.
# placeholder = st.empty()

# dataframe filter 



# df['version'] = pd.to_datetime(data['version'])
def getSimilarNodes(model,node):
    similarity=model.wv.most_similar(node)
    similar_nodes=pd.DataFrame()
    similar_nodes['Similar_Node']=[row[0] for i,row in enumerate(similarity)]
    similar_nodes['Similarity_Score']=[row[1] for i,row in enumerate(similarity)]
    similar_nodes['Source_Node']=node
    return similar_nodes

def flattenList(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list

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

# Static database graphs
tab1, tab2, tab3, tab4 = st.tabs(["Number of Submission Over Time by Field", 
                                  "Number of Submission by Country",
                                  "Number of Submission by Institute",
                                  "Network Graph by Author"])

with tab1:
    st.markdown("### Number of Submission Over Time by Field")
    # Filter
    paper_category = st.multiselect("Select the paper category", pd.unique(data['category']))
    df1 = data.loc[data['category'].isin(paper_category)]
    
    fig = px.histogram(df1, x="category")
    st.write(fig)
    
with tab2: 
    st.markdown("### Number of Submission by Country")
    # Filter
    country_category = st.multiselect("Select the countires", pd.unique(data['Country']))
    df2 = data.loc[data['Country'].isin(country_category)]
    
    fig2 = px.histogram(data_frame = df2, x = 'Country')
    st.write(fig2)
    
with tab3:
    st.markdown("### Number of Submission by Institute")
    
    institute_category = st.multiselect("Select the institution", pd.unique(data['Institute']))
    df3 = data.loc[data['Institute'].isin(institute_category)]
    
    fig3 = px.histogram(data_frame = df3, x = 'Institute')
    st.write(fig3)
    
with tab4:

    # #########################################################
    # paper_category_2 = st.multiselect("Select the field", pd.unique(data['category']))
    # records=db.read_text("arxiv-metadata-oai-snapshot.json").map(lambda x:json.loads(x))
    # ai_docs = (records.filter(lambda x:any(ele in x['categories'] for ele in paper_category_2)==True))
    # get_metadata = lambda x: {'id': x['id'],
    #               'title': x['title'],
    #               'category':x['categories'],
    #               'abstract':x['abstract'],
    #              'version':x['versions'][-1]['created'],
    #                      'doi':x["doi"],
    #                      'authors_parsed':x['authors_parsed']}

    # df4=ai_docs.map(get_metadata).to_dataframe().compute()
    # #########################################################
    st.markdown("### Network Graph by Author")
    
    # # Filter
    paper_category_2 = st.multiselect("Select the field", pd.unique(data['category']))
    # df4 = data.loc[data['category'].isin(paper_category_2)]

    if len(paper_category_2) == 0:
       st.text('Please choose at least 1 field to get started')
    # # Concatenate the author first and last names
    # df4['num_authors']=df4['authors_parsed'].apply(lambda x:len(x))
    # df4['authors']=df4['authors_parsed'].apply(lambda authors:[(" ".join(author)).strip() for author in authors])
    # df4['author_pairs']=df4['authors'].apply(lambda x:list(combinations(x, 2)))

    # field_authors=pd.DataFrame(flattenList(df4['authors'].tolist())).rename(columns={0:'authors'})
    # papers_by_authors=field_authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False)
    # # Keeping Authors who have published more than 2 Papers
    # nodes_to_keep=papers_by_authors.loc[papers_by_authors['Number of Papers Published']>2,'authors'].tolist()
    # # Generating the Edges of the Co-Author Network¶
    # authors_pairs=df4['author_pairs'].tolist()
    # authors_edge_list=[item for sublist in authors_pairs for item in sublist]
    # authors_weighted_edge_list=list(Counter(authors_edge_list).items())
    # authors_weighted_edge_list=[(row[0][0],row[0][1],row[1]) for idx,row in enumerate(authors_weighted_edge_list)]  
    # G1=nx.Graph()
    # G1.add_weighted_edges_from(authors_weighted_edge_list)
    # sub_g=nx.subgraph(G1,nodes_to_keep)
    # G=nx.Graph(sub_g)
    # isolated_node=nx.isolates(G)
    # G.remove_nodes_from(list(nx.isolates(G)))

    # num_sampling=10
    # random_walks=[]
    # length_of_random_walk=10
    # for node in tqdm(G.nodes(),desc="Iterating Nodes"):
    #     for i in range(0,num_sampling):
    #         random_walks.append(getRandomWalk(G,node,length_of_random_walk))

    # deepwalk_model = Word2Vec()
    # deepwalk_model.build_vocab(random_walks)
    # deepwalk_model=Word2Vec(window=5, sg=1,negative=5,vector_size=128,epochs=20,compute_loss=True)
    # deepwalk_model.save("deepwalk1.model")

    # ai_authors=pd.DataFrame(flattenList(df4['authors'].tolist())).rename(columns={0:'authors'})
    # papers_by_authors=ai_authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False)
    # coauthor_nodes=getCoAuthorshipNetwork(G,papers_by_authors['authors'].tolist()[4:10])
    # coauthor_subgraph=nx.subgraph(G,coauthor_nodes)
    # print(papers_by_authors)

    # nx.write_gexf(coauthor_subgraph, "CoAuthor_Subgraph_Author4to10.gexf")
    # pyvis_nt=Network(height='800px', width='100%',heading='')
    # pyvis_nt.from_nx(coauthor_subgraph)
    # pyvis_nt.save_graph(f'pyvis_graph.html')

    # fig4 = open(f'pyvis_graph.html', 'r', encoding='utf-8')
    # components.html(fig4.read(), height=435)

    # Take too long to run, use a saved visualization for demonstration purpose
    # HtmlFile = open("Author4to10_CoAuthorGraph.html", 'r', encoding='utf-8')
    # fig4 = HtmlFile.read() 
    # components.html(fig4)
    fig4 = Image.open('graph.png')

    st.image(fig4, caption='Network Graph')
   
    

