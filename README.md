# Analysis and Prediction of Poker Hand Strength

This project involves the analysis and prediction of poker hand strength using UCI Poker Hand data set:

https://archive.ics.uci.edu/ml/datasets/Poker+Hand

## Data Preparation

Firstly, there is no missing data. Secondly, RandomForestClassifer is applied to get the feature importance and, as explained
in the uploaded report as well, there is no feature that we can ignore as the lowest feature importance is approximately 7% 
which also makes sense since we can't separate the suites from the card ranks. 

Lastly, StandardScaler (which standardizes the features by removing mean and scaling to unit variance) 
is applied to transform the data in a way that is easier for our Neural Network to converge.

## Data Modeling

The data after transformation, is fed into the MLP (Multi-Layer Perceptron) classifier followed by tons of other classifiers in order
to present a comparison between their performance measure (mainly accuracy). The list of all classifiers is given below:

- **Bagging**
- **Random Forest**
- **AdaBoost**
- **KNN**
- **Naïve Bayes**
- **Decision Tree**
- **Linear SVM**
- **OutputCodeClassifier with Linear SVM**
- **OneVsAll with Linear SVM**
- **Multi-Layer Perceptron**

### Note 
Detailed explanation of each Data Science step (applied) is present in the report (PDF) uploaded. The data exploration, comparison table,
classification report etc. are all given in the report. This description is just the gist of the project.
