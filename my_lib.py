import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imblearn

from sklearn.model_selection import learning_curve        


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                       n_jobs=1, train_sizes=np.linspace(.1,1.0,8)):
    #plot Setup
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    #unpack plot data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_std, test_std = [np.std(A, axis=1) for A in [train_scores, test_scores]]
    train_mean, test_mean = [np.mean(A, axis=1) for A in [train_scores, test_scores]]

    plt.grid()
    
    #overlay std_dev on train curve
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    
    #overlay std_dev on validation curve
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    
    #plot train points
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    
    #plot validation points
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def topn(gridsearch, n):
    GS_df = pd.DataFrame(gridsearch.cv_results_)
    topn = GS_df.sort_values('mean_test_score',ascending=False).head(n)
    return topn

def read_test_results(file):
    #Reads Test Results into pandas DataFrame
    with open(file) as f:
        content = f.readlines()
        content = content[1:]
        df = [list(map(float, line.strip().split())) for line in content]
        n = len(df) + 1
        df = pd.DataFrame(df, index=range(1,n), columns=['Reward','Steps'])
    return df

def read_train_results(file):
    #Reads training_log_file
    with open(file) as f:
        content = f.readlines()
        df = []
        previous_ep = 100
        # eps, reward, steps
        accumulate = [0]*3
        batch = 0
        for line in content:
            if 'reward' in line:
                ep, reward, steps = line.split()[1::2]
                ep, reward, steps = int(ep[:-1]), float(reward[:-1]), int(steps)
                if ep < previous_ep:
                    df.append(accumulate)
                    accumulate = [0]*3
                #inc ep count
                accumulate[0]+=1
                accumulate[1]+=reward
                accumulate[2]+=steps
                
                previous_ep = ep
    n = len(df)
    df = pd.DataFrame(df, index=range(n))[1:-1]
    df[1] = df[1]/df[0]
    df[2] = df[2]/df[0]
    df.columns = ['eps', 'mean_reward', 'mean_steps']
    return df
