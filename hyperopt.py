from skopt import gp_minimize
from functools import partial

x_lst = []
score_lst = []


#in ORDER [(learning rate),[epochs]]
space = [(-5,4)] 

def my_function(*args):
    x = args[0][0]
    global final
    final = (x**4)+(3*x**3)-(9*x**2)-23*x-12
    
    return final

def print_result(params):
    

    x_lst.append(params.x_iters[-1][0])
    score_lst.append(params.func_vals[-1])
    


res = gp_minimize(my_function, space, n_calls=10, 
                  random_state=0, callback=print_result,
                 verbose = True)

print("Best parameters: ", res.x, "with best score: ", -res.fun)
