import pandas as pd
import matplotlib.pyplot as plt

def plot_profile(df,text,h=3,title=None):
    plt.rcParams["figure.figsize"] = (20,h)
    p = df.plot.bar(x='char')
    p = plt.xticks(rotation='horizontal',fontsize=1+round(60*20/len(text)))
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
