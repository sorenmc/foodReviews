# foodReviews

## Resume
Predicting if a text review was negative or positive. [Data can be found here][1]

There are 1883 examples. The data was split into 70% training, 15% validation and 15% test.
Bayesian optimization from the framework "bayesian-optimization" and 10-fold cross validation
was used to tune hyper parameters.

[Scikit-learns Logistic regression][2] was used as a baseline model, since it is one of the simplest models. 
The only hyper parameter optimizable is the inverse regularization strength, 'C'.
Best value that was found after 20 evaluations of bayesian optimization with 10-fold cross validation
was C = 0.11 which evaluated to 87.2 % mean accuracy. This resulted in 84.4% validation accuracy and 84.5% test accuracy.



[1]: 
[2]: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html]
