# foodReviews

## Resume
Predicting if a text review was negative or positive. [Data can be found here][1]

There are 1883 examples. The data was split into 70% training, 15% validation and 15% test.
Bayesian optimization from the framework [GPyOpt][2] and 10-fold cross validation
was used to tune hyper parameters.

[Scikit-learns Logistic regression][3] was used as a baseline model, since it is one of the simplest models. 
The only hyper parameter optimizable is the inverse regularization strength, 'C'.
Best value that was found after 20 evaluations of bayesian optimization with 10-fold cross validation
was C = 4.575 which evaluated to 85.8 % mean accuracy. This resulted in 83.7% validation accuracy and 84.5% test accuracy.

We then tested the model with a regular neural network 



[1]: https://github.com/sorenmc/foodReviews/blob/master/data/dataset_examples.tsv
[2]: https://github.com/SheffieldML/GPyOpt
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
