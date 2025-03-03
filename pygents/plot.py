import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pygents.text import *

def matrix_plot(row_labels, col_labels, matrix, absmax, title = None, vmin = None, vmax = None, dpi = None, titlefontsize = None, width = 20):
    plt.rcParams["figure.figsize"] = (width,len(row_labels)/4)
    if not dpi is None:
        plt.rcParams["figure.dpi"] = dpi
    p = sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels, 
                    vmin = -absmax if vmin is None else vmin, 
                    vmax = absmax if vmax is None else vmax, 
                    cmap='RdYlGn', annot=True)
    if title is not None:
        if titlefontsize is None:
            titlefontsize = 32 if len(title) < 50 else round(32 * 50 / len(title))
        p.set_title(title,fontsize = titlefontsize)
    plt.show()

def plot_profile(df,text,h=3,title=None):
    plt.rcParams["figure.figsize"] = (20,h)
    p = df.plot.bar(x='char')
    p = plt.xticks(rotation='horizontal',fontsize=1+round(60*20/len(text)))
    if title is not None:
        plt.title(title)

def plot_hbar(df,labels,values,title=None):
    plt.rcParams["figure.figsize"] = (20,round(len(df)/5))
    p = df[[labels,values]].plot.barh(x=labels); p.invert_yaxis();
    if title is not None:
        plt.title(title)

def plot_hbars(df,labels,values,title=None):
    plt.rcParams["figure.figsize"] = (20,round(len(df)/5))
    p = df[[labels]+values].plot.barh(x=labels); p.invert_yaxis();
    if title is not None:
        plt.title(title)

def plot_bars(df,labels,values,title=None,fontsize=None):
    plt.rcParams["figure.figsize"] = (20,3)
    p = df[[labels]+values].plot.bar(x=labels);
    p = plt.xticks(rotation='horizontal',fontsize = 1+round(60*20/len(df)) if fontsize is None else fontsize)
    if title is not None:
        fontsize = 32 if len(title) < 50 else round(32 * 50 / len(title))
        plt.title(title,fontsize = fontsize)

def plot_dict(dic,labels,values,title=None,head=None):
    df = pd.DataFrame([(key, dic[key]) for key in dic],columns=[labels,values])
    df.sort_values(values,ascending=False,inplace=True)
    if head is not None:
        df = df[:head]
    plt.rcParams["figure.figsize"] = (20,round(len(df)/5))
    p = df[[labels,values]].plot.barh(x=labels); p.invert_yaxis();
    if title is not None:
        plt.title(title)

def plot_dict_bars(dic,labels,values,title=None,head=None,dim=(8,5)):
    df = pd.DataFrame([(key, dic[key]) for key in dic],columns=[labels,values])
    #df.sort_values(values,ascending=False,inplace=True)
    if head is not None:
        df = df[:head]
    if not dim is None:
        plt.rcParams["figure.figsize"] = dim
    p = df[[labels,values]].plot.bar(x=labels); #p.invert_yaxis();
    if title is not None:
        plt.title(title)

def plot_profile_probabilities(counters,text,max_n,plot=True,debug=False):
    df = pd.DataFrame(profile_probabilities(counters,text,max_n,debug=debug),columns=['pos','char','p+','p-'])
    #print(df)
    df.set_index('pos',inplace=True)
    if plot:
        plt.rcParams["figure.figsize"] = (20,3)
        plot_profile( df[['char','p+']],text)
        plot_profile( df[['char','p-']],text)
    return df

def plot_profile_freedoms(model,text,max_n,plot=True,debug=False):
    df = pd.DataFrame(profile_freedoms(model,text,max_n,debug=debug),columns=['pos','char','f+','f-'])
    #print(df)
    df.set_index('pos',inplace=True)
    if plot:
        plt.rcParams["figure.figsize"] = (20,3)
        plot_profile( df[['char','f+']],text)
        plot_profile( df[['char','f-']],text)
    return df

def plot_profile_avg_freedom(model,text,n_min,n_max,col):
    sdf = None
    for n in range(1,7):
        plt.rcParams["figure.figsize"] = (20,2)
        df = profile_freedoms_df(model,text,n,debug=False)
        df[col] = df[col] / df[col].max()
        plot_profile(df[['char',col]],text,title=str(n)+col)
        if sdf is None:
            sdf = df[['char',col]]
        else:
            sdf[col] = sdf[col] + df[col]

    sdf[col] = sdf[col] / sdf[col].max()
    plot_profile(sdf[['char',col]],text,title='1-7 '+col)


def subplots_hist(df,labels,bins=100,fontsize=20):
    fig, axs = plt.subplots(1, len(labels), figsize=(20, 4), sharey=True)
    for ax, label in zip(axs,labels):
        #p = ax.hist(df[[label]],bins=bins); p = ax.set_xlabel(label)
        p = ax.hist(df[label],bins=bins); p = ax.set_xlabel(label,fontsize=fontsize)


# graph plots

# https://plotly.com/python/tree-plots/
# https://igraph.org/python/doc/tutorial/tutorial.html

from treelib import Node, Tree

import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import math



def make_annotations(pos, text, M, font_size=20, font_color='rgb(0,0,250)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        scaled_font_size = round(font_size/math.sqrt(len(text[k])))
        if scaled_font_size < 2:
            scaled_font_size = 2
        annotations.append(
            dict(
                text=text[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=2*M-pos[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=scaled_font_size),
                showarrow=False)
        )
    return annotations

def igraph_draw(labels,edges,title):
    #https://igraph.org/python/doc/tutorial/tutorial.html
    G = Graph()
    #G = Graph(directed=True) # TODO!!!???
    nr_vertices = len(labels)
    G.add_vertices(nr_vertices)
    G.add_edges(edges)

#    lay = G.layout('rt')
    lay = G.layout('tree')

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   ))
    fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  mode='markers',
                  name='bla',
                  marker=dict(symbol='circle',#symbol='circle-dot',
                                size=18, #18
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  text=labels,
                  hoverinfo='text',
                  opacity=0.8
                  ))
    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

    fig.update_layout(title=title,
              annotations=make_annotations(position, labels, M),
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
              )
    fig.show()



def dict2graph(dictree):
    parents_children_dict = {}
    for child in dictree: 
        dictcount(parents_children_dict,dictree[child])
    parents_children_list = list(parents_children_dict)
    parents_children_list.sort(key=lambda i: len(i), reverse=True)
    nodes = []
    edges = []
    nodes2index = {}
    for child in parents_children_list:
        index = len(nodes)
        nodes.append(child)
        nodes2index[child] = index
        if child in dictree:
            edges.append((nodes2index[child],nodes2index[dictree[child]]))
    for child in dictree: 
        if not child in parents_children_list:
            index = len(nodes)
            nodes.append(child)
            nodes2index[child] = index
            edges.append((nodes2index[child],nodes2index[dictree[child]]))
    return nodes, edges
assert str(dict2graph({'c1':'c0','c2':'c0'})) == "(['c0', 'c1', 'c2'], [(1, 0), (2, 0)])"


def dict2tree(dictree,debug=False):
    parents_children_dict = {}
    children = set()
    for child in dictree: 
        dictcount(parents_children_dict,dictree[child])
        children.add(child)
    parents_children_list = list(parents_children_dict)
    parents_children_list.sort(key=lambda i: len(i), reverse=True)
    if debug:
        print(parents_children_list)
    tree = Tree()
    tree.create_node('', '') # root
    for child in parents_children_list:
        if debug:
            print(child,'->',dictree[child] if child in dictree else '')
        if not child in dictree:
            tree.create_node(child, child, parent='')
        else:
            tree.create_node(child, child, parent=dictree[child])
    for child in children - parents_children_dict.keys(): 
        if debug:
            print(child,'->',dictree[child] if child in dictree else '')
        if not child in parents_children_list:
            tree.create_node(child, child, parent=dictree[child])
    return tree
#print(dict2tree({'c1':'c0','c2':'c0','c3':'c2'}))
assert(str(dict2tree({'c1':'c0','c2':'c0','c3':'c2'},False)).replace('\n','')) == "└── c0    ├── c1    └── c2        └── c3"

