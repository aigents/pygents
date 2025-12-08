# MIT License
# 
# Copyright (c) 2015-2025 Aigents®, Anton Kolonin, Anna Arinicheva 
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

from pygents.util import dictcount, agg_min_max_avg_mpe
from pygents.plot import matrix_plot, plot_bar_from_list
from pygents.aigents_api import TextMetrics, punct, Learner
import random
import pandas as pd

def df2labeled(df,binary=False):
    data = []
    for _, row in df.iterrows():
        # Text identification: first, check the 2nd column; if NaN, take the text from the 1st column
        text = row.iloc[1] if pd.notna(row.iloc[1]) else row.iloc[0]
        if not binary:
            primary_distortion = row.iloc[2]  # The primary cognitive distortion from the 3rd column
            secondary_distortion = row.iloc[3] if pd.notna(row.iloc[3]) else None  # The secondary distortion from the 4th column, if present
        else:
            primary_distortion = 'Distortion' if row.iloc[2] != 'No Distortion' else 'No Distortion'
            secondary_distortion = None
        cats = (primary_distortion,) if secondary_distortion is None else (primary_distortion,secondary_distortion)
        data.append((text, tuple([c.replace(' ','_') for c in cats])))
    return data

def language_metrics(lang,metrics_list,path):
    metrics = {}
    for m in metrics_list:
        metrics[m] = path + lang + '/' + m + '.txt'
    return metrics

def dictval(dic,key,val):
    return dic[key] if key in dic else val 

def our_evaluator_test(all_metrics,expected_distortions,text,threshold):
    dic = {}
    for m in all_metrics:
        dic[m] = True if (m in expected_distortions) else False
    return dic

def our_evaluator_tm(all_metrics,tm,text,threshold):
    metrics = tm.get_sentiment_words(text)
    dic = {}
    for m in all_metrics:
        dic[m] = True if m in metrics and metrics[m] > threshold else False
    return dic

def our_evaluator_top(all_metrics,tm,text,threshold,top=2):
    metrics = tm.get_sentiment_words(text)
    ranked_metrics = sorted(metrics.items(), key=lambda x: (x[1],x[0]),reverse=True)[:top] # "stable sort"
    dic = {}
    for rm in ranked_metrics:
        m = rm[0]
        dic[m] = True if m in metrics and metrics[m] > threshold else False
    return dic

def our_evaluator_top1(all_metrics,tm,text,threshold,top=1):
    metrics = tm.get_sentiment_words(text)
    ranked_metrics = sorted(metrics.items(), key=lambda x: (x[1],x[0]),reverse=True)[:top] # "stable sort"
    dic = {}
    for rm in ranked_metrics:
        m = rm[0]
        dic[m] = True if m in metrics and metrics[m] > threshold else False
    return dic

def our_evaluator_true(all_metrics,tm,text,threshold):
    dic = {}
    for m in all_metrics:
        dic[m] = True
    return dic
    
def our_evaluator_false(all_metrics,tm,text,threshold):
    dic = {}
    for m in all_metrics:
        dic[m] = False
    return dic

def our_evaluator_random(all_metrics,tm,text,threshold):
    dic = {}
    for m in all_metrics:
        dic[m] = random.choice([True, False])
    return dic

def pre_rec_f1_from_counts(true_positive, true_negative, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0 

def evaluate_tm_df(df,tm,evaluator,threshold,all_metrics,debug=False):
    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}
    pre = {}
    rec = {}
    f1 = {}
    acc = {}
    for _, row in df.iterrows():
        # Text definition: first, check the 2nd column; if NaN, take the text from the 1st column.
        text = row.iloc[1] if pd.notna(row.iloc[1]) else row.iloc[0]
        primary_distortion = row.iloc[2]  # The main cognitive distortion from the 3rd column
        secondary_distortion = row.iloc[3] if pd.notna(row.iloc[3]) else 'No Distortion'  # The secondary distortion from the 4th column, if it exists
        ground_distortions = []
        if primary_distortion != 'No Distortion':
            ground_distortions.append(primary_distortion.replace(' ','_'))
        if secondary_distortion != 'No Distortion':
            ground_distortions.append(secondary_distortion.replace(' ','_'))

        if evaluator == our_evaluator_test:
            distortions_by_metric = evaluator(all_metrics,ground_distortions,text,threshold) #hack to test metrics
        else:
            distortions_by_metric = evaluator(all_metrics,tm,text,threshold)

        if debug:
            print(ground_distortions,'=>',[m for m in distortions_by_metric if distortions_by_metric[m]])
        
        for metric in distortions_by_metric:
            our_distortion = distortions_by_metric[metric]
            if (metric in ground_distortions) and our_distortion == True:
                dictcount(true_positives,metric)
            if (not metric in ground_distortions) and our_distortion == True:
                dictcount(false_positives,metric)
            if (not metric in ground_distortions) and our_distortion == False:
                dictcount(true_negatives,metric)
            if (metric in ground_distortions) and our_distortion == False:
                dictcount(false_negatives,metric)

    if debug:
        #print()
        print('TP:',true_positives)
        print('FP:',false_positives)
        print('TN:',true_negatives)
        print('FN:',false_negatives)
    
    for metric in all_metrics:
        precision, recall, f1score = pre_rec_f1_from_counts(dictval(true_positives,metric,0), dictval(true_negatives,metric,0), 
                                   dictval(false_positives,metric,0), dictval(false_negatives,metric,0))
        pre[metric] = precision
        rec[metric] = recall
        f1[metric] = f1score
        acc[metric] = (dictval(true_positives,metric,0) + dictval(true_negatives,metric,0)) / len(df)
    
    return pre, rec, f1, acc


def evaluate_metrics(tm, test_df, inclusion_threshold, detection_thresholds, name, all_metrics, n_max = 4, selection_metric = 'FN',
                     f1_score=False, all_scores = False, averages=False, evaluator=our_evaluator_tm, accumulator=None):
    all_metrics = sorted(all_metrics)
    pres = [[] for i in range(len(all_metrics))]
    recs = [[] for i in range(len(all_metrics))]
    f1s = [[] for i in range(len(all_metrics))]
    accs = [[] for i in range(len(all_metrics))]
    f1avgs = []
        
    for t in detection_thresholds:
        pre, rec, f1, acc = evaluate_tm_df(test_df,tm,evaluator,t/100.0,all_metrics,debug=False)
        mi = 0
        for metric in all_metrics:
            pres[mi].append(pre[metric])
            recs[mi].append(rec[metric])
            f1s[mi].append(f1[metric])
            accs[mi].append(acc[metric])
            mi += 1
        f1avg = sum([f1[metric] for metric in all_metrics])/len(all_metrics)
        f1avgs.append(f1avg)
        if not accumulator is None:
            accumulator.append((n_max,inclusion_threshold,selection_metric,t,f1avg,f1))

    if all_scores:
        matrix_plot(all_metrics, detection_thresholds, pres, 1.0, title = f'Precision(DT): DS={name}, SM={selection_metric}, IT={inclusion_threshold}, NM={n_max}', vmin = 0, vmax = 1.0, titlefontsize = 20, dpi = 300, width = 10)
        matrix_plot(all_metrics, detection_thresholds, recs, 1.0, title = f'Recall(DT): DS={name}, SM={selection_metric}, IT={inclusion_threshold}, NM={n_max}', vmin = 0, vmax = 1.0, titlefontsize = 20, dpi = 300, width = 10)
    if all_scores or f1_score:
        matrix_plot(all_metrics, detection_thresholds, f1s, 1.0, title = f'F1(DT): DS={name}, SM={selection_metric}, IT={inclusion_threshold}, NM={n_max}', vmin = 0, vmax = 1.0, titlefontsize = 20, dpi = 300, width = 10)
    if averages:
        plot_bar_from_list('Detection Threshold',detection_thresholds,'F1',f1avgs)


def evaluate_model(model, test_df, test_path, model_prefix, validation_fraction, inclusion_thresholds, detection_thresholds, 
                   n_max=4, selection_metrics = ('FN',), rescale=False, weighted=False, f1_score=False, all_scores=False, name='Multiclass', averages=False,
                   evaluator=our_evaluator_tm, accumulator=None):
    for inclusion_threshold in inclusion_thresholds:
        for selection_metric in selection_metrics:
            model.save(path=test_path, name=f'{model_prefix}-{inclusion_threshold}',metric=selection_metric,
                   inclusion_threshold=inclusion_threshold,rescale=rescale)
            all_metrics = model.export(metric=selection_metric, inclusion_threshold=inclusion_threshold,rescale=rescale).keys() - {'No_Distortion'}
            tm = TextMetrics(language_metrics('',all_metrics,path=test_path + f'/{model_prefix}-{inclusion_threshold}'),
                         encoding="utf-8",metric_logarithmic=True,weighted=weighted,debug=False)
            evaluate_metrics(tm,test_df,inclusion_threshold,detection_thresholds,name,all_metrics,
                                  n_max=n_max, selection_metric = selection_metric,f1_score=f1_score,all_scores=all_scores,
                                  averages=averages,evaluator=evaluator,accumulator=accumulator)



def full_test_circle(df, test_path, model_prefix, validation_fraction, inclusion_thresholds, detection_thresholds, 
                     n_max=4, selection_metrics = ('FN',), rescale=False, weighted=False, f1_score=False, all_scores=False, name='Multiclass', averages=False, 
                     split_shift=0, evaluator=our_evaluator_tm, accumulator=None):
    train_df = df[(df.index + split_shift) % validation_fraction != 0]
    test_df  = df[(df.index + split_shift) % validation_fraction == 0]
    print(f'Shift={split_shift}: train={len(train_df)}, test={len(test_df)}')

    learner = Learner(n_max=n_max)
    model = learner.learn(df2labeled(train_df),n_max=n_max,punctuation=punct,sent=True,debug=False)
    print('Labels count:', model.labels)

    evaluate_model(model,test_df,test_path,model_prefix,validation_fraction,inclusion_thresholds,detection_thresholds,
                                n_max=n_max,selection_metrics=selection_metrics, rescale=rescale, weighted=weighted, f1_score=f1_score,
                                all_scores=all_scores,name=name,averages=averages,evaluator=evaluator,accumulator=accumulator)


def summarize_full_test_circle(results):
    summary = []
    for l1,l2,l3 in zip(results[0],results[1],results[2]):
        summary.append( l1[0:4] + agg_min_max_avg_mpe((l1[4],l2[4],l3[4])))
    summary = sorted(summary, key=lambda x: (-x[6],x[2],x[0],-x[1],x[3])) # sort by -F1avg, SM, NM, -IT, RT
    NM, IT, SM, DT = summary[0][:4]
    F1 = summary[0][6]
    print(f'NM={NM}, IT={IT}, SM={SM}, DT={DT}, F1={F1}')
    return summary
 
