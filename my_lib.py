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
    