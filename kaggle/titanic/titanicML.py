import pandas
from sklearn import linear_model

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
# initialize the algorithm.
alg = linear_model.LogisticRegression(random_state=1)
# train the algorithm
alg.fit(train[predictors], train["Survived"])
# predictions for test set.
predictions = alg.predict(test[predictors])
# dataframe for submission with PassengerId and Survived columns.
submission = pandas.DataFrame({"PassengerId": test["PassengerId"],
                               "Survived": predictions})
# output
submission.to_csv("testSurvivors.csv", index=False)
