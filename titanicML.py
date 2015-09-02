import pandas
from sklearn.ensemble import RandomForestClassifier

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

# information used to determine whether the passenger survived.
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends.
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
# train the algorithm
alg.fit(train[predictors], train["Survived"])
# predictions for test set.
predictions = alg.predict(test[predictors])
# dataframe for submission with PassengerId and Survived columns.
submission = pandas.DataFrame({"PassengerId": test["PassengerId"],
                               "Survived": predictions})
# output
submission.to_csv("testSurvivors.csv", index=False)
