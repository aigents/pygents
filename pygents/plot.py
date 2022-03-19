import pandas as pd
import matplotlib.pyplot as plt

def plot_profile(df,text,h=3,title=None):
    plt.rcParams["figure.figsize"] = (20,h)
    p = df.plot.bar(x='char')
    p = plt.xticks(rotation='horizontal',fontsize=1+round(60*20/len(text)))
    if title is not None:
        plt.title(title)

def plot_bar(df,labels,values,title=None):
    plt.rcParams["figure.figsize"] = (20,round(len(df)/5))
    p = df[[labels,values]].plot.barh(x=labels); p.invert_yaxis();
    if title is not None:
        plt.title(title)

def plot_dict(dic,labels,values,title=None,head=None):
    df = pd.DataFrame([(key, dic[key]) for key in dic],columns=[labels,values])
    df.sort_values(values,ascending=False,inplace=True)
    if head is not None:
        df = df[:head]
    plt.rcParams["figure.figsize"] = (20,round(len(df)/5))
    p = df[[labels,values]].plot.barh(x=labels); p.invert_yaxis();
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

    sdf[c] = sdf[col] / sdf[col].max()
    plot_profile(sdf[['char',col]],text,title='1-7 '+col)
    

