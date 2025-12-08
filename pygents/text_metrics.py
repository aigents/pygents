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

psy_metrics = {'en':[
('😊','positive','Positive statements','Emotionally positive sentiment about situation or context'),
('😟','negative','Negative statements','Emotionally negtative sentiment about situation or context'),
('🎭','contradictive','Contradictive statements','Opposition of positive and negative references to the same situation or context'),
('😳','rude','Rude statements','Emotionally rude (obscene vocabulary) expressions about situation or context'),
('😱','catastrophizing','Catastrophizing','Exaggerating the importance of negative events'),
('⚖️','dichotomous-reasoning','Dichotomous Reasoning','Thinking that an inherently continuous situation can only fall into two categories'),
('👎','disqualifying-positive','Disqualifying the Positive','Unreasonably discounting positive experiences'),
('💔','emotional-reasoning','Emotional Reasoning','Thinking that something is true based on how one feels, ignoring the evidence to the contrary'),
('🔮','fortune-telling','Fortune-telling','Making predictions, usually negative ones, about the future'),
('🏷','labeling','Labeling and mislabeling','Labeling yourself or others while discounting evidence that could lead to less disastrous conclusions'),
('🔬','magnification','Magnification and Minimization','Magnifying negative aspects or minimizing positive aspects'),
('🙅','mental-filtering','Mental Filtering','Paying too much attention to negative details instead of the whole picture'),
('😎','mindreading','Mindreading','Believing you know what others are thinking'),
('📉','overgeneralizing','Overgeneralizing','Making sweeping negative conclusions based on a few examples'),
('🙇🏼','personalizing','Personalizing','Believing yourself or others are behaving negatively because of oneself, without considering more plausible or external explanations for behavior'),
('📌','should-statement','Should statements','Having a fixed idea on how you and/or others should behave')],
'ru':[
('😊','positive','Позитивные высказывания','Эмоционально позитивное отношение к ситуации или контексту'),
('😟','negative','Негативные высказывания','Эмоционально негативное мнение о ситуации или контексте'),
('🎭','contradictive','Противоречивые высказывания','Противопоставление положительных и отрицательных упоминаний об одной и той же ситуации или контексте'),
('😳','rude','Грубые высказывания','Эмоционально грубые (нецензурная лексика) выражения о ситуации или контексте'),
('😱','catastrophizing','Катастрофизация','Преувеличение важности негативных событий'),
('⚖️','dichotomous-reasoning','Дихотомическое мышление','Мысль о том, что непрерывная по своей сути ситуация может делиться только на две категории'),
('👎','disqualifying-positive','Дисквалификация позитива','Необоснованное обесценивание позитивного опыта'),
('💔','emotional-reasoning','Эмоциональное мышление','Думать, что что-то верно, основываясь на своих чувствах, игнорируя доказательства обратного'),
('🔮','fortune-telling','Гадание','Предсказания будущего, обычно негативные'),
('🏷','labeling','Навешивание ярлыков','Навешивание ярлыков на себя и других, при этом игнорируя доказательства, которые могли бы привести к менее катастрофическим выводам'),
('🔬','magnification','Преувеличение и преуменьшение','Усиление отрицательных аспектов или минимизация положительных аспектов'),
('🙅','mental-filtering','Ментальная фильтрация','Слишком много внимания уделяется негативным деталям, а не всей картине'),
('😎','mindreading','Чтение мыслей','Вера в то, что вы знаете, о чем думают другие'),
('📉','overgeneralizing','Чрезмерное обобщение','Делать радикальные отрицательные выводы на основе нескольких примеров'),
('🙇🏼','personalizing','Персонализация','Убеждение, что я или другие ведут себя негативно по своей натуре, без рассмотрения более правдоподобных или внешних объяснений поведения'),
('📌','should-statement','Предписывание','Наличие фиксированного представления о том, как вам и/или другим следует вести себя')]}

def psy_metric_lists(metrics,lang='en',emojis=False):
    psy_lang_metrics = psy_metrics[lang]
    keys = []
    vals = []
    for m in psy_lang_metrics:
        if m[1] in metrics:
            keys.append(m[0]+m[2] if emojis else m[2])
            vals.append(metrics[m[1]])
    return (keys, vals)
    
def psy_metric_texts(metrics,lang='en',emojis=True,extended=False,lists=None,markup=False):
    psy_lang_metrics = psy_metrics[lang]
    texts = []
    for m in psy_lang_metrics:
        if m[1] in metrics:
            s = m[0]
            t = m[2]
            v = metrics[m[1]]
            if emojis: # multiple emojis
                s = s+s+s+s if v >= 0.75 else s+s+s if v > 0.5 else s+s if v > 0.25 else s
            summary = (s+t+' - '+m[3] if extended else s+t)
            if not lists is None and m[1] in lists:
                features = lists[m[1]]
                features = ["__"+" ".join(f)+"__" for f in features] if markup else [" ".join(f) for f in features]
                summary += " : " + str(", ".join(features))
            texts.append(summary)
    return texts
