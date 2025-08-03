# Supplementary Code and material for paper "Aligning Numerical Data Quantification with Human Intuition" 
1. quantification_2025_run.ipynb - code for Quantification on Synthetic Data and empirical calculation and usage of proposed metric "SC+".
2. S&P_2025_dataset_quantification.ipynb - code for Quantification on S&P 2021-2025 dataset. Shows the usage of metric SC, NCDC an SC+. Results proposes alignment with Human Intuition and the limitation of SC+ and limitations of other metrics used in the situation when SC<0.65
3. quantification_2025_questions.pdf and quantification_2025_answers.pdf - List of questions and answers (responses) to validate unsupervised quantification of distributions produced using quantification_2025_run.ipynb code on synthetic data.
4. License - MIT License file for free/open source usage.

## Data

SP_500_2021_2025.csv

The data file represents S&P 500 data from 2021 to 2025, consisting of (Date, Open, High, Low, Close, Volume) columns representing Date  and trading datapoints Open price, Highest price, Lowest price, Closing price and Volume of S&P 500 stocks on the particular date.
