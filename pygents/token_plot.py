from pygents.token import *
from pygents.util import list2matrix 
from pygents.plot import matrix_plot 


def evaluate_freedom_tokenizer_options(test_texts,ref_tokenizer,tokenizer,ngram_params,thresholds,plot=True,title=None,nospaces=False,debug=False):
    rlist = []
    for nlist in ngram_params:
        for threshold in thresholds: 
            tokenizer.set_options(nlist = nlist, threshold=threshold)
            avg_f1 = evaluate_tokenizer_f1(test_texts,ref_tokenizer,tokenizer,nospaces=nospaces,debug=False)
            if debug:
                print(nlist,threshold,avg_f1)
            rlist.append((nlist,threshold,avg_f1))
    if plot:
        r,c,m = list2matrix(rlist)
        matrix_plot(r,c,m,1.0,title,vmin=0.0)
    return rlist

