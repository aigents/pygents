import pandas as pd
import numpy as np
import math
from collections import defaultdict

from util import dictcount, dict_compress_with_loss, dictdict_div_dict, dictdict_mul_dictdict
from a_api import tokenize_re, punct, build_ngrams

punct = punct + "”“–&•"


def count_ngrams_basic(df, n_max: int, binary = False, clean_punct=True):

    distortions = defaultdict(int)

    # Creating dictionaries for counting n-grams
    n_gram_dicts = defaultdict(lambda: defaultdict(int))  # A dictionary for each distortion (distortion-n-gram-n_gram_frequency)
    all_n_grams = defaultdict(int)  # A general dictionary for all n-grams
    
    uniq_n_gram_dicts = defaultdict(lambda: defaultdict(int)) # Counts of uniq N-grams by Distortion
    uniq_all_n_grams = defaultdict(int)  # A general dictionary for all n-grams uniq by text
    n_gram_distortions = defaultdict(lambda: defaultdict(int)) # Counts of distortiions by N-gram

    # Loop through the rows of the DataFrame
    for _, row in df.iterrows():
        # Text identification: first, check the 2nd column; if NaN, take the text from the 1st column
        text = row.iloc[1] if pd.notna(row.iloc[1]) else row.iloc[0]
        if not binary:
            primary_distortion = row.iloc[2]  # The primary cognitive distortion from the 3rd column
            secondary_distortion = row.iloc[3] if pd.notna(row.iloc[3]) else None  # The secondary distortion from the 4th column, if present
        else:
            primary_distortion = 'Distortion' if row.iloc[2] != 'No Distortion' else 'No Distortion'
            secondary_distortion = None
        
        dictcount(distortions,primary_distortion)
        if secondary_distortion:
            dictcount(distortions,secondary_distortion)
        
        # Text tokenization
        tokens = [t for t in tokenize_re(text) if not (t in punct or t.isnumeric())] if clean_punct else tokenize_re(text)

        # Generation and counting of n-grams (from 1 to 4)
        for n in range(1, n_max + 1):
            n_grams = build_ngrams(tokens, n)
            dictcount(all_n_grams, n_grams)
            dictcount(n_gram_dicts[primary_distortion], n_grams)  # Increment the counter for the corresponding primary distortion
            if secondary_distortion:
                dictcount(n_gram_dicts[secondary_distortion], n_grams) # Increment the counter for the corresponding secondary distortion (if present)

            uniq_n_grams = set(n_grams)
            for uniq_n_gram in uniq_n_grams:
                dictcount(uniq_n_gram_dicts[primary_distortion], uniq_n_gram)
                dictcount(uniq_all_n_grams, uniq_n_gram)
                dictcount(n_gram_distortions[uniq_n_gram],primary_distortion)
                if secondary_distortion:
                    dictcount(uniq_n_gram_dicts[secondary_distortion], uniq_n_gram)
                    dictcount(n_gram_distortions[uniq_n_gram],secondary_distortion)
                
    # Normalizing distortion-specific counts by total counts
    norm_n_gram_dicts = {}
    for n_gram_dict in n_gram_dicts:
        norm_n_gram_dict = {}
        norm_n_gram_dicts[n_gram_dict] = norm_n_gram_dict
        dic = n_gram_dicts[n_gram_dict]
        for n_gram in dic:
            #print(dic[n_gram])
            #print(all_n_grams[n_gram])
            #break
            if len(n_gram) <= n_max:
                norm_n_gram_dict[n_gram] = float( dic[n_gram] ) / all_n_grams[n_gram]

    # Normalize uniq counts 
    norm_uniq_n_gram_dicts = {}
    for uniq_n_gram_dict in uniq_n_gram_dicts: # iterate over all distortions
        norm_uniq_n_gram_dict = {}
        norm_uniq_n_gram_dicts[uniq_n_gram_dict] = norm_uniq_n_gram_dict
        dic = uniq_n_gram_dicts[uniq_n_gram_dict] # pick uniq count of ngrams per distortion
        nonuniq_dic = n_gram_dicts[uniq_n_gram_dict] # pick non-uniq count of ngrams per distortion - BUG!?
        # Normalize uniq Document counts of N-grams by distortion by Documents count by Distortion
        for n_gram in dic:
            if len(n_gram) <= n_max:
                #norm_uniq_n_gram_dict[n_gram] = float( dic[n_gram] ) * nonuniq_dic[n_gram] / distortions[uniq_n_gram_dict] / len(n_gram_distortions[n_gram]) / all_n_grams[n_gram]
                # divide (uniq count if ngrams by distorion) by (count of texts with given distorion) and (count of ngrams with given distortion)
                norm_uniq_n_gram_dict[n_gram] = float( dic[n_gram] ) / distortions[uniq_n_gram_dict] / len(n_gram_distortions[n_gram])
 
    n_gram_distortions_counts = {}
    for n_gram, dist_dict in n_gram_distortions.items():
        n_gram_distortions_counts[n_gram] = len(dist_dict)

    return distortions, n_gram_dicts, all_n_grams, norm_n_gram_dicts, uniq_n_gram_dicts, uniq_all_n_grams, n_gram_distortions, \
    norm_uniq_n_gram_dicts, n_gram_distortions_counts


def count_ngrams_plus(df, n_max: int, binary = False, clean_punct=True):
    N = len(df)
    distortions, n_gram_dicts, all_n_grams, norm_n_gram_dicts, uniq_n_gram_dicts, uniq_all_n_grams, n_gram_distortions, \
    norm_uniq_n_gram_dicts, n_gram_distortions_counts = count_ngrams_basic(df, n_max, binary = binary, clean_punct=clean_punct)

    norm = dictdict_div_dict(n_gram_dicts,all_n_grams)
    norm_uniq = dictdict_div_dict(uniq_n_gram_dicts,uniq_all_n_grams)
    norm_norm_uniq = dictdict_mul_dictdict(norm,norm_uniq)
    norm_norm_uniq_norm = dictdict_div_dict(norm_norm_uniq,n_gram_distortions_counts)
    
    # norm_norm_uniq_norm_norm = norm_norm_uniq_norm[dist][n_gram] * n_gram_distortions[n_gram][dist] / distortions[dist]
    norm_norm_uniq_norm_norm = {} # looks like desired magic
    for dist in distortions:
        if not dist in norm_norm_uniq_norm: #hack
            continue
        dic = norm_norm_uniq_norm[dist]
        norm_norm_uniq_norm_norm[dist] = {}
        for n_gram in dic:
            norm_norm_uniq_norm_norm[dist][n_gram] = dic[n_gram] * n_gram_distortions[n_gram][dist] / distortions[dist]

    nl_mi = {}
    for dist in uniq_n_gram_dicts:
        dic = uniq_n_gram_dicts[dist]
        nl_mi[dist] = {}
        for n_gram in dic:
            nl_mi[dist][n_gram] = dic[n_gram] * dic[n_gram] / (distortions[dist] * uniq_all_n_grams[n_gram])
    
    fcr = {}
    cfr = {}
    mr = {}
    for dist in uniq_n_gram_dicts:
        dic = uniq_n_gram_dicts[dist]
        fcr[dist] = {}
        cfr[dist] = {}
        mr[dist] = {}
        for n_gram in dic:
            features_by_cat = sum(dic.values()) # features by category
            cats_by_feature = sum(n_gram_distortions[n_gram].values()) # categories by feature
            fcr[dist][n_gram] = dic[n_gram] / cats_by_feature # feature to category relevance - denominated by n of categories by feature
            cfr[dist][n_gram] = dic[n_gram] / features_by_cat # category to feature relevance - denominated by n of features by category
            mr[dist][n_gram] = dic[n_gram] * dic[n_gram] / (features_by_cat * cats_by_feature)
    
    return distortions, n_gram_dicts, all_n_grams, norm_n_gram_dicts, uniq_n_gram_dicts, uniq_all_n_grams, n_gram_distortions, \
    norm_uniq_n_gram_dicts, n_gram_distortions_counts, norm, norm_uniq, norm_norm_uniq, norm_norm_uniq_norm, norm_norm_uniq_norm_norm, \
    fcr, cfr, mr, nl_mi, N
