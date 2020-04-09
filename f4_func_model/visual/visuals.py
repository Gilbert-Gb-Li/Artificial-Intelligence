###########################################
# Suppress matplotlib user warnings
###########################################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split


def plot(data_train):
    """
    连续型数据
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    rm = data_train['Age']
    medv = data_train['Survived']
    plt.scatter(rm, medv, c=medv)

    rm = data_train['Fare']
    medv = data_train['Survived']
    plt.scatter(rm, medv, c='b')
    plt.show()



def ModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training titanic.
        The learning and validation scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10, 7))

    # Create three different models based on max_depth
    for k, depth in enumerate([1, 3, 6, 10]):
        
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth=depth)

        # Calculate the training and testing scores
        sizes, train_scores, valid_scores = learning_curve(regressor, X, y,
            cv=cv, train_sizes=train_sizes, scoring='r2')
        
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis=1)
        train_mean = np.mean(train_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color='r', label='Training Score')
        ax.plot(sizes, valid_mean, 'o-', color='g', label='Validation Score')
        ax.fill_between(sizes, train_mean - train_std,
            train_mean + train_std, alpha=0.15, color='r')
        ax.fill_between(sizes, valid_mean - valid_std,
            valid_mean + valid_std, alpha=0.15, color='g')
        
        # Labels
        ax.set_title('max_depth = %s' % depth)
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('r2_score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad=0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and validation errors rates are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, valid_scores = validation_curve(DecisionTreeRegressor(), X, y,
        param_name="max_depth", param_range=max_depth, cv=cv, scoring='r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(max_depth, valid_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(max_depth, train_mean - train_std,
        train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(max_depth, valid_mean - valid_std,
        valid_mean + valid_std, alpha=0.15, color='g')
    
    # Visual aesthetics
    plt.legend(loc='lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('r2_score')
    plt.ylim([-0.05, 1.05])
    plt.show()


def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting titanic. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the titanic
        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.2, random_state=k)
        
        # Fit the titanic
        reg = fitter(X_train, y_train)
        
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        # Result
        print("Trial {}: ${:,.2f}".format(k+1, pred))

    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))
