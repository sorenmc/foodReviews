import modelClass
from bayes_opt import BayesianOptimization

def main():
    dataModel = modelClass.modelClass()
    dataModel.loadData()

    
    optimizer = BayesianOptimization(
        f = dataModel.trainModelLR,
        pbounds = {'C':(0.1,1000)})
    optimizer.maximize(init_points=3, n_iter= 20)
    print("Best hyper parameters", optimizer.max)
if __name__ == "__main__":
    main()