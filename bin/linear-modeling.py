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

import numpy as np
from scipy.stats import norm
from scipy.special import expit as logistic
from scipy.optimize import minimize

MAX_ITERATIONS = 100
TOLERANCE = 1e-5
REGULARIZER = 1

def initialize_weights(training_data, initialization_type="zero", gaussian_mean=0, gaussian_sd=0.2):
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
    
def perform_regression(data, weights, regression_type="logistic"):
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
            this_pred  = predictions[i, 0]
            this_label = labels[i, 0]
            this_dot = dot_product[i, 0]
            pdf  = norm.pdf(dot_product[i, 0])

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

def evaluate_accuracy(
        predictions,
        labels,
        regression_type="logistic"):
    """ Evaluate performance of current weights 
    
    regression_type = <logistic, probit>
    """
    
    if regression_type == "multiclass":
        prediction_max = np.argmax(predictions, axis=1)
        label_max   = np.argmax(labels, axis=1)
        accuracy = np.sum(prediction_max == label_max) * 100.0 / predictions.shape[0]
        
    elif regression_type == "logistic" or regression_type == "probit":
        #if predictions.ndim == 2:
        #    predictions = predictions[:,0]
        predictions[predictions > 0.5] = 1.0
        predictions[predictions < 1.0] = 0.0
        accuracy = np.sum(predictions == labels) * 100.0 / predictions.shape[0]
    
    return accuracy
    
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
    accuracy = evaluate_accuracy(
        predictions=regression_output,
        labels=test_labels,
        regression_type=regression_type)
    
    return accuracy, regression_output

def func_probit(weights):
    """
    """
    
    regression_output, dot_product = perform_regression(
        data=training_data, 
        weights=weights, 
        regression_type="probit")
    log_output = np.log(regression_output + TOLERANCE)
    result = -1 * np.sum(
        np.multiply(training_labels, log_output) + np.multiply(1 - training_labels, 1 - log_output)
    ) + np.sum(weights**2)
    
    return result

def func_probit_gradient(weights):
    """
    """
    
    updated_weights = weights[:, np.newaxis]
    regression_output, dot_product = perform_regression(
        data=training_data, 
        weights=updated_weights, 
        regression_type="probit")
    gradient = calculate_gradient(
        data=training_data, 
        labels=training_labels, 
        predictions=regression_output, 
        dot_product=dot_product, 
        weights=updated_weights, 
        regression_type="probit")
    
    return gradient[:,0]

def newton_solver(
        training_data, training_labels, 
        test_data, test_labels, 
        initialization_type="zero", 
        regression_type="logistic",
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
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
            regression_type=regression_type)
        
        # Calculate Hessian matrix
        hessian = calculate_hessian(
            data=training_data, 
            labels=training_labels,
            predictions=regression_output, 
            dot_product=dot_product, 
            regression_type=regression_type)
        
        # Update weights
        updated_weights = update_weights(
            weights=weights,
            gradient=gradient,
            hessian=hessian)
        
        # Calculate difference between weights
        difference = np.linalg.norm(updated_weights - weights)
        
        # Calculate accuracy of predictions
        accuracy, regression_output = test_weights(
            updated_weights=updated_weights, 
            test_data=test_data, 
            test_labels=test_labels, 
            regression_type=regression_type)
        weights = updated_weights
        
        # Report results
        print("\tIteration: " + str(i+1) + ", Change in weights: " + str(round(difference, 3)) + ", Test accuracy: " + str(round(accuracy, 5)))
        
        # Check if critical level reached, and exit if True
        if difference < tolerance:
            print("Training complete.")
            break

def bfgs_solver(
        training_data, training_labels, 
        test_data, test_labels, 
        initialization_type="zero", 
        regression_type="logistic",
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        gaussian_mean=0, 
        gaussian_sd=0.2):
    """ Perform regression MAP estimation with a BFGS solver 
    
    regression_type = <logistic, probit>
    """
    
    # Initialize weights
    weights = initialize_weights(
        training_data, 
        initialization_type=initialization_type,
        gaussian_mean=gaussian_mean, 
        gaussian_sd=gaussian_sd)
    
    # For each iteration up to max
    print("Performing MAP estimation with BFGS solver...")
    
    if regression_type == "probit":
        x0 = weights[:,0]
        probit_params = minimize(
            func_probit, 
            x0, 
            jac=func_probit_gradient, 
            method='BFGS', 
            tol=TOLERANCE, 
            options={
                'disp': True, # Print convergence messages
                'maxiter': MAX_ITERATIONS
            })
        weights[:,0] = probit_params.x
   
    # Calculate accuracy of predictions
    accuracy, regression_output = test_weights(
            updated_weights=weights, 
            test_data=test_data, 
            test_labels=test_labels, 
            regression_type=regression_type)
    
    # Report results
    print("\tTest accuracy: " + str(round(accuracy, 5)))