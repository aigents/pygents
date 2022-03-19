import matplotlib.pyplot as plt

def plot_profile(df,text,h=3,title=None):
    plt.rcParams["figure.figsize"] = (20,h)
    p = df.plot.bar(x='char')
    p = plt.xticks(rotation='horizontal',fontsize=1+round(60*20/len(text)))
    if title is not None:
        plt.title(title)

