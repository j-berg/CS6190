"""linear-modeling summary

Examples:
    # Logistic regression with weight-zero initialization
    newton_solver(
        training_data=training_data, 
        training_labels=training_labels, 
        test_data=test_data, 
        test_labels=test_labels, 
        initialization_type="zero", 
        regression_type="logistic")
        
    # Logistic regression with weight-Gaussian initialization
    newton_solver(
        training_data=training_data, 
        training_labels=training_labels, 
        test_data=test_data, 
        test_labels=test_labels, 
        initialization_type="random", 
        regression_type="logistic")
        
    # Logistic regression with weight-Gaussian initialization & high st-dev
    newton_solver(
        training_data=training_data, 
        training_labels=training_labels, 
        test_data=test_data, 
        test_labels=test_labels, 
        initialization_type="random", 
        regression_type="logistic",
        gaussian_sd=1)
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import expit as logistic
from scipy.optimize import minimize

MAX_ITERATIONS = 100
TOLERANCE = 1e-5
REGULARIZER = 1

def initialize_weights(
        training_data, 
        initialization_type="zero", 
        gaussian_mean=0, 
        gaussian_sd=0.2):
    """ Initialize weights 
    type = "zero" or "random"
    """
    
    print("Initializing " + str(initialization_type) + " weights...")
    if initialization_type == "zero":
        weights = np.zeros(
            training_data.shape[1])
    elif initialization_type == "random":
        weights = np.random.normal(
            loc=gaussian_mean, 
            scale=gaussian_sd, 
            size=training_data.shape[1])
        
    return weights[:, np.newaxis]
    
def perform_regression(
        data, 
        weights, 
        regression_type="logistic"):
    """
    
    regression_type = <logistic, probit>
    """
    
    dot_product = np.matmul(data, weights)
    
    if regression_type == "logistic":
        regression_output = logistic(dot_product)
        
    elif regression_type == "probit":
        regression_output = norm.cdf(dot_product)

    return regression_output, dot_product

def calculate_gradient(
        data, 
        labels, 
        predictions, 
        dot_product, 
        weights, 
        regression_type="logistic", 
        regularizer=REGULARIZER, 
        tolerance=TOLERANCE):
    """ Calculate gradient from predictions
    
    regression_type = <logistic, probit>
    """
    if regression_type == "logistic" or regression_type == "multiclass":
        gradient = np.matmul(data.T, predictions - labels)
        
    elif regression_type == "probit":
        eye_matrix = np.eye(predictions.shape[0])
        for i in range(predictions.shape[0]):
            eye_matrix[i, i] = (norm.pdf(dot_product[i, 0]) 
                             / (predictions[i, 0] * (1 - predictions[i, 0]) 
                             + TOLERANCE))
        
        gradient = np.matmul(
            np.matmul(data.T, eye_matrix), 
            predictions - labels)

    # Update gradient
    gradient += weights / regularizer
    
    return gradient

def calculate_hessian(
        data, 
        labels, 
        predictions, 
        dot_product, 
        regression_type="logistic", 
        regularizer=REGULARIZER):
    """ Calculate Hessian matrix
    
    regression_type = <logistic, probit>
    """
    
    eye_matrix = np.eye(predictions.shape[0])
    
    if regression_type == "logistic":
        for i in range(predictions.shape[0]):
            eye_matrix[i, i] = predictions[i, 0] * (1 - predictions[i, 0])
            
    elif regression_type == "probit":
        for i in range(predictions.shape[0]):

            t1 = (1 
                / (predictions[i, 0] 
                * (1 - predictions[i, 0]) 
                + TOLERANCE))
            t2 = (
                (predictions[i, 0] - labels[i, 0]) 
                / (predictions[i, 0]**2 
                * (1 - predictions[i, 0]) 
                + TOLERANCE))
            t3 = (
                (predictions[i, 0] - labels[i, 0]) 
                / ((1- predictions[i, 0])**2 
                * predictions[i, 0] 
                + TOLERANCE))
            t4 = (
                (predictions[i, 0] - labels[i, 0]) 
                * dot_product[i, 0] / (predictions[i, 0] 
                * (1 - predictions[i, 0]) 
                * norm.pdf(dot_product[i, 0]) 
                + TOLERANCE))

            eye_matrix[i, i] = (t1 - t2 + t3 - t4) * (norm.pdf(dot_product[i, 0])**2)

    # Perform regularization
    hessian = np.matmul(np.matmul(data.T, eye_matrix), data) + np.eye(data.shape[1]) / regularizer
    
    return hessian

def update_weights(
        weights,
        gradient,
        hessian):
    """ Update weights given the gradient and Hessian matrices
    """

    updated_weights = weights - np.matmul(np.linalg.inv(hessian), gradient)
    
    return updated_weights

def evaluate_mse(
        predictions,
        labels):
    """ Evaluate performance of current weights 
    
    regression_type = <logistic, probit>
    """
    
    mse = 1 / predictions.shape[0] * np.sum(predictions - labels)**2

    return mse
    
def evaluate_correlation(
        predictions,
        labels):
    """
    """
    
    corr_matrix = pd.DataFrame(np.concatenate((predictions, labels), axis=1))
    corr = corr_matrix.corr()
    R_sq = corr.iloc[-1,-2]**2
    
    return R_sq

def test_weights(
        updated_weights, 
        test_data, 
        test_labels, 
        regression_type="logistic"):
    """ Evaluate current weights on test data
    
    regression_type = <logistic, probit>
    """
    
    # Run regression with current weights
    regression_output, dot_product = perform_regression(
        data=test_data, 
        weights=updated_weights, 
        regression_type=regression_type)
    
    # Report accuracy
    mse = evaluate_mse(
        predictions=regression_output,
        labels=test_labels)
    
    r_corr = evaluate_correlation(
        predictions=regression_output,
        labels=test_labels)

    return mse, r_corr, regression_output

def newton_solver(
        training_data, training_labels, 
        test_data, test_labels, 
        initialization_type="zero", 
        regression_type="logistic",
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        regularizer=REGULARIZER,
        gaussian_mean=0, 
        gaussian_sd=0.2):
    """Perform regression MAP estimation with a Newton-Raphson solver 
    
    regression_type = <logistic, probit>
    """
    
    # Initialize weights
    weights = initialize_weights(
        training_data, 
        initialization_type=initialization_type,
        gaussian_mean=gaussian_mean, 
        gaussian_sd=gaussian_sd)
    
    # For each iteration up to max
    print("Performing MAP estimation with Newton-Raphson solver...")
    mse_record = []
    for i in range(max_iterations): 
        
        # Perform regression prediction
        regression_output, dot_product = perform_regression(
            data=training_data, 
            weights=weights, 
            regression_type=regression_type)
        
        # Evaluate using gradient descent
        gradient = calculate_gradient(
            data=training_data, 
            labels=training_labels, 
            predictions=regression_output, 
            dot_product=dot_product, 
            weights=weights, 
            regression_type=regression_type,
            regularizer=REGULARIZER)
        
        # Calculate Hessian matrix
        hessian = calculate_hessian(
            data=training_data, 
            labels=training_labels,
            predictions=regression_output, 
            dot_product=dot_product, 
            regression_type=regression_type,
            regularizer=REGULARIZER)
        
        # Update weights
        updated_weights = update_weights(
            weights=weights,
            gradient=gradient,
            hessian=hessian)
        
        # Calculate difference between weights
        difference = np.linalg.norm(updated_weights - weights)
        
        # Calculate accuracy of predictions
        mse, r_corr, regression_output = test_weights(
            updated_weights=updated_weights, 
            test_data=test_data, 
            test_labels=test_labels, 
            regression_type=regression_type)
        weights = updated_weights
        
        # Report results
        print(
            "\tIteration: " + str(i+1) 
            + ", Change in weights: " + str(round(difference, 3)) 
            + ", MSE: " + str(round(mse, 3))
            + ", R_sq: " + str(round(r_corr, 3)))
        mse_record.append(mse)
        
        # Check if critical level reached, and exit if True
        if difference < tolerance:
            print("Training complete.")
            return weights, mse_record
    
    if difference > tolerance:
        print("Training was unable to complete.")
        np.zeros(1), [np.inf]
        
    