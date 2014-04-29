#ifndef LOGISTIC_REG_H_
#define LOGISTIC_REG_H_

#include "boost/numeric/ublas/matrix.hpp"

#include <iostream>

class LogisticRegression
{
public:
	// Compute the sigmoid value for a scalar
	// Params: Input data in a scalar
	// Return: The sigmoid value
	double Sigmoid(double data);

	// Compute the element-wise sigmoid function
	// Params: Input data in a matrix
	// Return: Apply the elements-wise operation and return
	boost::numeric::ublas::matrix<double> Sigmoid(boost::numeric::ublas::matrix<double> data);

	// Load the training data into a matrix from a delimit-separated format
	// Params: Path to the training data in a delimit-separated format
	// Return: 1 if something goes wrong
	int LoadTrainDataFromDelimit(const std::string path, const char delimit);

	// Load the test data into a matrix from a delimit-separated format
	// Params: Path to the training data in a delimit-separated format
	// Return: 1 if something goes wrong
	int LoadTestDataFromDelimit(const std::string path, const char delimit);

	// Normalize the features in the train_data
	void NormalizeFeatures();

	// Add one column of 1s to the data matrix (as the intercept)
	void AddInterceptColumn(boost::numeric::ublas::matrix<double> & data);

	// Add one column of 1s to the train_data matrix (as the intercept)
	void AddInterceptColumn();

	// Gradient descent to find the best fit
	// Params: the learning rate alpha, and the number of iterations
	// Return: the vector holding the costs through the iterations
	std::vector<double> GradientDescent(const float alpha, const int max_iteration);

	// Gradient descent to find the best fit, with regularization
	// Params: the learning rate alpha, the maximum number of iterations, and the regularization term lambda
	// Return: the vector holding the costs through the iterations
	std::vector<double> GradientDescentWithRegularization(const float alpha, const int max_iteration, const double lambda);

	// Compute the costs function value in each gradiant descent iteration
	double ComputeCosts();

	// Compute the costs function value in each gradiant descent iteration, with regularization
	// Params: the regularization term lambda
	// Return: the costs
	double ComputeCosts(const double lambda);

	// Predict a single case using the theta learned from gradient descent
	// Params: the single point data for the prediction
	// Return: the predicted value
	double Predict(std::vector<double> data);

	// Predict multiple cases using the theta learned from gradient descent
	// Params: the data for the prediction
	// Return: the predicted value
	std::vector<double> Predict(boost::numeric::ublas::matrix<double> data);

	// Predict the test_data using the theta learned from gradient descent
	// Params: N/A
	// Return: vector of the predicted value
	std::vector<double> Predict();

	// Binary classify the test_data using the theta learned from the train_data
	// Params: N/A
	// Return: vector of the predicted class label
	std::vector<double> BinaryClassify();

	// Evaluate the predictions with the test_target
	void EvaluatePredictions(const std::vector<double> predictions);

	// Evaluate the classification labels with the test_target
	void EvaluateClassificationLabels(const std::vector<double> labels);

	// Print more information when the debug mode is ON
	void PrintDebugInfo();

	LogisticRegression();
	~LogisticRegression();

private:
	std::string train_data_path;
	boost::numeric::ublas::matrix<double> train_data;
	boost::numeric::ublas::matrix<double> train_data_mean;
	boost::numeric::ublas::matrix<double> train_data_stddev;
	boost::numeric::ublas::matrix<double> target;
	boost::numeric::ublas::matrix<double> theta;

	std::string test_data_path;
	boost::numeric::ublas::matrix<double> test_data;
	boost::numeric::ublas::matrix<double> test_target;

	// Threshold for the convergence of the gradient descent process
	double epsilon;

};

#endif // End of LOGISTIC_REG_H
