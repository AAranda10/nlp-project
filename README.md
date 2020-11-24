# Classification of Programming Languages Used by GitHub Repo

## Introduction
For this project, we will be scraping data from Github repository README files and build a model that can predict what programming language a repository is using, given the text of the README file. The deliverables are:<br>
    - A well documented jupyter notebook that contains our analysis.<br>
    - Google slides that summarize our findings.<br>

## Goals
To predict the programming language used in a repository, based on the text of its README file.<br>

## Key Findings and Takeaways

![takeaways](https://github.com/AAranda10/nlp-project/blob/main/Summary.png)

- Wordcloud is used to visualize and compare the relative frequency of different keywords using in the programming language JavaScript and Python. 
- Hyperparameter tuning has been performed for feature extraction and we found the min_df has an strong impact on both accuracy and overfitting. 
- Seven classification algorithms have been evaluted and we found LogisticRegression outperforms the other six. 
- The accuracy of modeling on test is 69%, which beats the baseline by 10%. 

## Documents in the Repo
- README.md:
    - Overview of the zillow clustering project 
- .gitignore:
    - files to be ignored. 
- Jupyter notebooks:
    - final_notebook.ipynb
        - a well documented notebook that contains our analysis.
- src:
    - acquire.py: 
        - functions to scrape the README files from Github Repositories
    - prepare.py:
        - functions used for basic cleaning, tokenizing, lemmatizing and stopwords removal.
    - model.py:
        - functions to compute the classification metrics and perform polynominal models. 

## Data Dictionary

| Column | Description |
| --- | ---|
| label | the programming language used by the reository |
| text | the original text in the README file in the repository |
| lemmatized | the wrangled and lemmatized text |
| clean | the wrangled, leammtized and stopwords-removed text |
| words | a list of words split from the clean text |

## Future Work
1. To obtain more training datasets to minimize the overfitting. 
2. To expand our binary classification model to multiclass classifiction.
