import modelClass
#from bayes_opt import BayesianOptimization
import GPyOpt



def main():
    dataModel = modelClass.modelClass()
    dataModel.loadDataSequence()
    
    domain = [{'name': 'nCNN','type':'discrete','domain':tuple(range(1,6))},
              {'name': 'nDense','type':'discrete','domain':tuple(range(0,3))},
              {'name': 'nEmbedding','type':'discrete','domain':tuple(range(5,200))},
              {'name': 'nCNNFilters','type':'discrete','domain':tuple(range(2,1000))},
              {'name': 'nNNFilters','type':'discrete','domain':tuple(range(3,1000))},
              {'name': 'nKernel','type':'discrete','domain':tuple(range(1,4))},
              {'name': 'nStrides','type':'discrete','domain':tuple(range(1,2))},
              {'name': 'poolSize','type':'discrete','domain':tuple(range(1,2))}]
    optimizer = GPyOpt.methods.BayesianOptimization(
        dataModel.optimizeCNN,
        bounds
        )
    max_iter = 20
    optimizer.run_optimization(max_iter)
    print(optimizer.x_opt)
    
    
if __name__ == "__main__":
    main()