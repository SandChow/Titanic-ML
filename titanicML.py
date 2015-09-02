import pandas
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt

# import data into script.
train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

# fill in empty age columns with median age.
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# convert genders to numbers so that machine learning algorithms can be used. male = 0, female = 1.
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

# convert embarkation port names to numbers. 0 to S, 1 to C and 2 to Q.
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# fill missing fare columns with median fare.
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# generating a familysize column
train["FamilySize"] = train["SibSp"] + train["Parch"]

# the .apply method generates a new series
train["NameLength"] = train["Name"].apply(lambda x: len(x))


# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
titles = train["Name"].apply(get_title)
titlesForTest = test["Name"].apply(get_title)
# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v
    titlesForTest[titlesForTest == k] = v
# Add in the title column.
train["Title"] = titles
test["Title"] = titlesForTest

# information used to determine whether the passenger survived.
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# best predictors
bestPredictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
alg.fit(train[bestPredictors], train["Survived"])
predictions = alg.predict(test[bestPredictors])

# dataframe for submission with PassengerId and Survived columns.
submission = pandas.DataFrame({"PassengerId": test["PassengerId"],
                               "Survived": predictions})
# output
submission.to_csv("testSurvivors.csv", index=False)

