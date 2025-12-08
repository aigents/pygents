# MIT License
# 
# Copyright (c) 2015-2025 Aigents®, Anton Kolonin 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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


def evaluate_freedom_tokenizer_multimetrics(test_texts,ref_tokenizer,tokenizer,ngram_params,thresholds,plot=True,title=None,nospaces=False,crossmetrics=False,test_counts=None,debug=False):
    rlist = []
    for nlist in ngram_params:
        for threshold in thresholds: 
            tokenizer.set_options(nlist = nlist, threshold=threshold)
            avg_f1, compratio, entropy = evaluate_tokenizer_f1_compratio_entropy(test_texts,ref_tokenizer,tokenizer,nospaces=nospaces,text_counts=test_counts,debug=False)
            if debug:
                print(nlist,threshold,avg_f1,compratio,entropy)
            rlist.append((nlist,threshold,avg_f1,compratio,entropy))
            #crlist.append((nlist,threshold,compratio))
    if plot:
        r,c,m = list2matrix([(i[0],i[1],i[2]) for i in rlist]) # F1 - F-score
        matrix_plot(r,c,m,1.0,'F1:'+title,vmin=0.0)
        r,c,m = list2matrix([(i[0],i[1],i[3]) for i in rlist]) # C% - Compression percentage
        matrix_plot(r,c,m,1.0,'C%:'+title,vmin=0.0,vmax=0.5)
        r,c,m = list2matrix([(i[0],i[1],i[4]) for i in rlist]) # ~S - Normalized anti-entropy
        matrix_plot(r,c,m,1.0,'~S:'+title,vmin=0.0,vmax=0.5)
        if crossmetrics:
            r,c,m = list2matrix([(i[0],i[1],(i[3]+i[4])/2) for i in rlist]) # C% - Compression percentage
            matrix_plot(r,c,m,1.0,'C%+~S:'+title,vmin=0.0,vmax=0.5)
        if crossmetrics:
            r,c,m = list2matrix([(i[0],i[1],(i[3]*i[4])) for i in rlist]) # C% - Compression percentage
            matrix_plot(r,c,m,1.0,'C%*~S:'+title,vmin=0.0,vmax=0.25)
    return rlist
