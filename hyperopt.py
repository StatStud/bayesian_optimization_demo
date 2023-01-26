from skopt import gp_minimize
from functools import partial

x_lst = []
y_lst = []
score_lst = []

model_lst = []
batch_lst = []

model = "bert-base-case"
batch = 16

#in ORDER [(learning rate),[epochs]]
space = [(0.00001, 0.001), 
         (2, 5)] 

def my_function(*args):
    x = args[0][0]
    y = args[0][1]
    global final
    final = (x - 2)**2 + (y - 3)**2
    
    return -final

def print_result(params):
    

    x_lst.append(params.x_iters[-1][0])
    y_lst.append(params.x_iters[-1][1])
    score_lst.append(params.func_vals[-1])
    
    model_lst.append(model)
    batch_lst.append(batch)


res = gp_minimize(my_function, space, n_calls=3, 
                  random_state=0, callback=print_result,
                 verbose = True)

print("Best parameters: ", res.x, "with best score: ", -res.fun)
