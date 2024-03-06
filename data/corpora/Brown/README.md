# BROWN CORPUS

A Standard Corpus of Present-Day Edited American
English, for use with Digital Computers.

by W. N. Francis and H. Kucera (1964)
Department of Linguistics, Brown University
Providence, Rhode Island, USA

Revised 1971, Revised and Amplified 1979

http://www.hit.uib.no/icame/brown/bcm.html

Distributed with the permission of the copyright holder,
redistribution permitted.

The text above is downloaded from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
The link above is obtained from http://www.nltk.org/nltk_data/

## FILES:

- brown_nolines.txt - downloaded from: http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt
- brown_nolines_en.txt - blank lines removed from brown_nolines.txt
- brown_nolines_ru.txt - translated to Russian from English with locale 'ru' from brown_nolines_en.txt
- brown_nolines_zh.txt - translated to Chinese (Simplified, People's Republic of China) from English with locale 'zh-cn' from brown_nolines_en.txt

Translations above are done with procedure in https://github.com/aigents/pygents/blob/main/notebooks/nlp/translation/googletrans.ipynb 

## REFERENCES:

- https://www.nltk.org/nltk_data/ (p.102)
- http://icame.uib.no/brown/bcm.html

## TODO

- remove double spaces
- remove everything between hash tags: 'grep "\#" brown_nolines_en.txt'
- remove < and > symbols ?
- remove | ?
- replace _ with spaces ?
- remove @ ?
- put sentence breaks in place of space after period ("\. ") except single letter before "." (" [a-zA-Z]\. ") and Dr/Mr/Mrs/Jr/Sr/Sen/La/Va/Dept/Calif/...






