#include "logistic_reg.h"
#include "utilz/utilz.h"

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

// Compute the sigmoid value for a scalar
// Params: Input data in a scalar
// Return: The sigmoid value
double LogisticRegression::Sigmoid(double data)
{
  return (1.0 / (1.0 + exp(-data)));
}

// Compute the sigmoid function
// Params: Input data in the form of a matrix or a vector
// Return: Return its sigmoid; for matrix and vector, do the elements-wise operation
boost::numeric::ublas::matrix<double> LogisticRegression::Sigmoid(boost::numeric::ublas::matrix<double> data)
{
  namespace bnu = boost::numeric::ublas;

  bnu::matrix<double> sigmoid_data = bnu::matrix<double> (data.size1(), data.size2());

  for (int i = 0; i < data.size1(); i++)
  {
    for (int j = 0; j < data.size2(); j++)
    {
      sigmoid_data(i, j) = (1.0 / (1.0 + exp(-data(i, j))));
    }
  }

  return (sigmoid_data);
}

// Load the training data into a matrix from a delimit-separated format
// Params: Path to the training data in a delimit-separated format
// Return: 1 if something goes wrong
int LogisticRegression::LoadTrainDataFromDelimit(const String path, const char delimit)
{
  namespace bnu = boost::numeric::ublas;

	train_data_path = path;

	std::ifstream input_file(train_data_path, std::ifstream::in);

	if (!input_file.is_open())
	{    
    std::cout << "Failed to open delimit-separated file: " << path << std::endl;
    return 1;
  }

  String line;
  getline(input_file, line);

  std::vector<VectorOfString> data;
  
  while (input_file.good())
  {
    VectorOfString fields = Split(line, delimit);
    data.push_back(fields);
    getline(input_file, line);
  }

  input_file.close();

  train_data = bnu::matrix<double> (data.size(), data[0].size() - 1);
  target = bnu::matrix<double> (data.size(), 1);

  for (int i = 0; i < data.size(); i++)
  {
    for (int j = 0; j < data[i].size() - 1; j++)
    {
      train_data.insert_element(i, j, ConvertStringToDouble(data[i][j]));
    }

    target.insert_element(i, 0, ConvertStringToDouble(data[i][data[i].size() - 1]));
  }
	return 0;
}

// Load the test data into a matrix from a delimit-separated format
// Params: Path to the test data in a delimit-separated format
// Return: 1 if something goes wrong
int LogisticRegression::LoadTestDataFromDelimit(const String path, const char delimit)
{
  namespace bnu = boost::numeric::ublas;

  test_data_path = path;

  std::ifstream input_file(test_data_path, std::ifstream::in);

  if (!input_file.is_open())
  {    
    std::cout << "Failed to open delimit-separated file: " << path << std::endl;
    return 1;
  }

  String line;
  getline(input_file, line);

  std::vector<VectorOfString> data;
  
  while (input_file.good())
  {
    VectorOfString fields = Split(line, delimit);
    data.push_back(fields);
    getline(input_file, line);
  }

  input_file.close();

  test_data = bnu::matrix<double> (data.size(), data[0].size() - 1);
  test_target = bnu::matrix<double> (data.size(), 1);

  for (int i = 0; i < data.size(); i++)
  {
    for (int j = 0; j < data[i].size() - 1; j++)
    {
      test_data.insert_element(i, j, ConvertStringToDouble(data[i][j]));
    }

    test_target.insert_element(i, 0, ConvertStringToDouble(data[i][data[i].size() - 1]));
  }
  return 0;
}


// Normalize the features in the train_data
void LogisticRegression::NormalizeFeatures()
{
  namespace bnu = boost::numeric::ublas;

  train_data_mean = bnu::zero_matrix<double> (1, train_data.size2());
  train_data_stddev = bnu::zero_matrix<double> (1, train_data.size2());

  for (int j = 0; j < train_data.size2(); j++)
  {
    bnu::matrix_column<bnu::matrix<double> > mc (train_data, j);
    for (int i = 0; i < mc.size(); i++)
    {
      train_data_mean(0, j) += (mc(i) / train_data.size1());
    }
  }

  for (int j = 0; j < train_data.size2(); j++)
  {
    bnu::matrix_column<bnu::matrix<double> > mc (train_data, j);
    for (int i = 0; i < mc.size(); i++)
    {
      train_data_stddev(0, j) += (pow(mc(i) - train_data_mean(0, j), 2) / train_data.size1());
    }
  }
  for (int j = 0; j < train_data_stddev.size2(); j++)
  {
    train_data_stddev(0, j) = sqrt(train_data_stddev(0, j));
  }

  for (int i = 0; i < train_data.size1(); i++)
  {
    for (int j = 0; j < train_data.size2(); j++)
    {
      train_data(i, j) = ((train_data(i, j) - train_data_mean(0, j)) / train_data_stddev(0, j));
    }
  }
}

// Add one column of 1s to the provided data matrix (as the intercept)
void LogisticRegression::AddInterceptColumn(boost::numeric::ublas::matrix<double> & data)
{
  namespace bnu = boost::numeric::ublas;

  bnu::matrix<double> ones (data.size1(), 1);

  for (int i = 0; i < ones.size1(); i++)
  {
    ones.insert_element(i, 0, 1.0);
  }

  data.resize(data.size1(), data.size2() + 1, true);
  
  for (int i = data.size2() - 1; i > 0; i--)
  {
    column(data, i) = column(data, i - 1);
  }
  column(data, 0) = column(ones, 0);
}

// Add one column of 1s to the train_data matrix (as the intercept)
void LogisticRegression::AddInterceptColumn()
{
  namespace bnu = boost::numeric::ublas;

  bnu::matrix<double> ones (train_data.size1(), 1);

  for (int i = 0; i < ones.size1(); i++)
  {
    ones.insert_element(i, 0, 1.0);
  }

  train_data.resize(train_data.size1(), train_data.size2() + 1, true);
  
  for (int i = train_data.size2() - 1; i > 0; i--)
  {
    column(train_data, i) = column(train_data, i - 1);
  }

  column(train_data, 0) = column(ones, 0);

}

// Gradient descent to find the best fit
// Params: the learning rate alpha, and the number of iterations
// Return: the vector holding the costs through the iterations
std::vector<double> LogisticRegression::GradientDescent(const float alpha, const int max_iteration)
{
  namespace bnu = boost::numeric::ublas;

  theta = bnu::zero_matrix<double> (train_data.size2(), 1);   // Initialize the theta matrix
  vector<double> costs_history;     // Initialize the vector to hold the costs through the iterations

  bnu::matrix<double> diff; 

  int iteration = 0;

  std::cout << "Starting gradient descent..." << std::endl;

  while (iteration < max_iteration) 
  {
    iteration++;

    // Octave: grad = X' * (sigmoid(X * theta) - y) ./ m;
    diff = bnu::prod(bnu::trans(train_data), (Sigmoid(bnu::prod(train_data, theta)) - target));
    theta -= diff * alpha / train_data.size1();
    
    double costs = ComputeCosts();

    // //debug
    std::cout << "Costs at iteration " << iteration << ": " << costs << std::endl;
    
    if ((!costs_history.empty()) && (std::abs(costs - costs_history.back()) / costs < epsilon))
    {
      // Reaches equilibrium; finish gradient descent and return
      costs_history.push_back(costs);

      std::cout << "Finished gradient descent at iteration " << iteration << std::endl;

      return (costs_history);
    }
    else
    {
      costs_history.push_back(costs);
    }
  }

  // Reaches maximum number of iterations; finish gradient descent and return
  std::cout << "Reached maxinum number of iterations." << std::endl;
  return costs_history;
}

// Gradient descent to find the best fit, with regularization
// Params: the learning rate alpha, the maximum number of iterations, and the regularization term lambda
// Return: the vector holding the costs through the iterations
std::vector<double> LogisticRegression::GradientDescentWithRegularization(const float alpha, const int max_iteration, const double lambda)
{
  namespace bnu = boost::numeric::ublas;

  theta = bnu::zero_matrix<double> (train_data.size2(), 1);   // Initialize the theta matrix
  vector<double> costs_history;     // Initialize the vector to hold the costs through the iterations

  //debug
  std::cout << "Initial costs: " << ComputeCosts(lambda) << std::endl;

  bnu::matrix<double> diff; 

  int iteration = 0;

  std::cout << "Starting gradient descent..." << std::endl;

  while (iteration < max_iteration) 
  {
    iteration++;

    // Octave: 
    // grad = X' * (sigmoid(X * theta) - y) ./ m; % Same as non-regularized version
    // grad(2,:) += theta(2,:) * lambda / m;

    diff = bnu::prod(bnu::trans(train_data), (Sigmoid(bnu::prod(train_data, theta)) - target));

    // Add the regularization terms
    for (int i = 1; i < diff.size1(); i++)
    {
      diff(i, 0) += (theta(i, 0) * lambda);
    }

    // Do the gradient descent on theta
    theta -= diff * alpha / train_data.size1();
    
    double costs = ComputeCosts(lambda);

    // //debug
    std::cout << "Costs at iteration " << iteration << ": " << costs << std::endl;
    
    if ((!costs_history.empty()) && (std::abs(costs - costs_history.back()) / costs < epsilon))
    {
      // Reaches equilibrium; finish gradient descent and return
      costs_history.push_back(costs);

      std::cout << "Finished gradient descent at iteration " << iteration << std::endl;

      return (costs_history);
    }
    else
    {
      costs_history.push_back(costs);
    }
  }

  // Reaches maximum number of iterations; finish gradient descent and return
  std::cout << "Reached maxinum number of iterations." << std::endl;
  return costs_history;
}

// Compute the costs function value in each gradiant descent iteration
double LogisticRegression::ComputeCosts()
{
  namespace bnu = boost::numeric::ublas;

  double costs = 0.0;

  // Octave:
  // m = size(X, 1);
  // for i = 1:m
  //   J += (- y(i) * log(sigmoid(X(i,:) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta))) / m;
  // endfor

  bnu::matrix<double> hypothesis = Sigmoid(bnu::prod(train_data, theta));

  for (int i = 0; i < train_data.size1(); i++)
  {
    costs += (-target(i, 0) * log(hypothesis(i, 0)) - (1 - target(i, 0)) * log(1 - hypothesis(i, 0))) / train_data.size1();
  }

  return costs;
}

// Compute the costs function value in each gradiant descent iteration, with regularization
// Params: the regularization term lambda
// Return: the costs
double LogisticRegression::ComputeCosts(const double lambda)
{
  namespace bnu = boost::numeric::ublas;

  double costs = 0.0;

  // Octave: 
  // m = size(X, 1);
  // for i = 1:m
  //   J += (- y(i) * log(sigmoid(X(i,:) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta))) / m;
  // endfor
  // n = size(theta, 1);
  // for j = 2:n % Regularization terms
  //   J += realpow(theta(j, 1), 2) * lambda / (2 * m);
  // endfor

  bnu::matrix<double> hypothesis = Sigmoid(bnu::prod(train_data, theta));

  for (int i = 0; i < train_data.size1(); i++)
  {
    costs += (-target(i, 0) * log(hypothesis(i, 0)) - (1 - target(i, 0)) * log(1 - hypothesis(i, 0))) / train_data.size1();
  }

  // Add the regularization terms
  for (int i = 1; i < theta.size1(); i++)
  {
    costs += pow(theta(i, 0), 2) * lambda / (2 * train_data.size1());
  }

  return costs;

}

// Predict a single case using the theta learned from gradient descent
// Params: the single point data for the prediction
// Return: the predicted value
double LogisticRegression::Predict(const std::vector<double> data)
{
  double prediction = theta(0, 0) * data[0];    // Initialize with the intercept

  for (int i = 1; i < data.size(); i++)
  {
    prediction += theta(i, 0) * ((data[i] - train_data_mean(0, i - 1)) / train_data_stddev(0, i - 1)); 
  }

  return (Sigmoid(prediction));
}

// Predict multiple cases using the theta learned from gradient descent
// Params: the data for the prediction
// Return: vector of the predicted value
std::vector<double> LogisticRegression::Predict(boost::numeric::ublas::matrix<double> data)
{
  // Add the intercept to the data matrix
  AddInterceptColumn(data);
  std::vector<double> predictions;

  for (int i = 0; i < data.size1(); i++)
  {
    double predict = theta(0, 0) * data(i, 0);    // Initialize with the intercept
    for (int j = 1; j < data.size2(); j++)
    {
      predict += theta(j, 0) * ((data(i, j) - train_data_mean(0, j - 1)) / train_data_stddev(0, j - 1)); 
    }

    predictions.push_back(Sigmoid(predict));
  }

  return predictions;
}

// Predict the test_data using the theta learned from gradient descent
// Params: N/A
// Return: vector of the predicted value
std::vector<double> LogisticRegression::Predict()
{
  // Add the intercept to the test_data matrix
  AddInterceptColumn(test_data);
  std::vector<double> predictions;

  for (int i = 0; i < test_data.size1(); i++)
  {
    double predict = theta(0, 0) * test_data(i, 0);    // Initialize with the intercept
    for (int j = 1; j < test_data.size2(); j++)
    {
      predict += theta(j, 0) * ((test_data(i, j) - train_data_mean(0, j - 1)) / train_data_stddev(0, j - 1)); 
    }

    predictions.push_back(Sigmoid(predict));
  }

  return predictions;
}

// Binary classify the test_data using the theta learned from the train_data
// Params: N/A
// Return: vector of the predicted class label
std::vector<double> LogisticRegression::BinaryClassify()
{
  // Add the intercept to the test_data matrix
  AddInterceptColumn(test_data);
  std::vector<double> class_labels;

  for (int i = 0; i < test_data.size1(); i++)
  {
    double predict = theta(0, 0) * test_data(i, 0);    // Initialize with the intercept
    for (int j = 1; j < test_data.size2(); j++)
    {
      predict += theta(j, 0) * ((test_data(i, j) - train_data_mean(0, j - 1)) / train_data_stddev(0, j - 1)); 
    }

    if (Sigmoid(predict) >= 0.5)
    {
      class_labels.push_back(1);
    }
    else
    {
      class_labels.push_back(0);
    }
  }
  return class_labels;
}

// Evaluate the predictions with the test_target
void LogisticRegression::EvaluatePredictions(const std::vector<double> predictions)
{
  std::cout << "Predict" << "\t\t" << "True" << "\t\t" << "Diff (in perc.)" << std::endl;
  for (int i = 0; i < predictions.size(); i++)
  {
    std::cout << predictions[i] << "\t\t" << test_target(i, 0) << "\t\t" << (abs(predictions[i] - test_target(i,0)) / test_target(i, 0) * 100) << "%" << std::endl;
  }
}

// Evaluate the classification labels with the test_target
void LogisticRegression::EvaluateClassificationLabels(const std::vector<double> labels)
{
  double correct = 0;

  std::cout << "Predict" << "\t\t" << "True" << std::endl;
  for (int i = 0; i < labels.size(); i++)
  {
    std::cout << labels[i] << "\t\t" << test_target(i, 0) << std::endl;

    if (labels[i] == test_target(i, 0))
    {
      correct++;
    }
  }

  std::cout << "Classification accuracy: " << correct / labels.size() << std::endl;
}

// Print more information when the debug mode is ON
void LogisticRegression::PrintDebugInfo()
{
  namespace bnu = boost::numeric::ublas;

  std::cout << "--------------- DEBUG INFO ---------------" << std::endl;
  
  std::cout << "Train data [1]: " << bnu::row(train_data, 0) << std::endl;
  std::cout << "Train target [1]: " << bnu::row(target, 0) << std::endl;
  std::cout << "Train data mean: " << train_data_mean << std::endl;
  std::cout << "Train data std. dev: " << train_data_stddev << std::endl;
  std::cout << "Test data [1]: " << bnu::row(test_data, 0) << std::endl;
  std::cout << "Test target [1]: " << bnu::row(test_target, 0) << std::endl;
  std::cout << "Theta: " << theta << std::endl;
  
  std::cout << "------------------------------------------" << std::endl;
}

LogisticRegression::LogisticRegression()
{
  // Threshold for the convergence of the gradient descent process
  epsilon = 1e-8;
}
	

LogisticRegression::~LogisticRegression()
{

}



int main (int argc, char* argv[]) 
{
  if (argc < 5) 
  { 
    // We expect at least 4 arguments: 
    // 1. Path to the data for training (required)
    // 2. The learning rate (required)
    // 3. The maximum number of iteration (required)
    // 4. Toggle regularization; lambda = 0 means no regularization
    // 5. Path to the data for testing (optional)
    // 6. Toggle debug mode (optional)
    std::cerr << "Usage: " << argv[0] << " [TRAIN_DATA] [LEARNING_RATE] [MAX_ITERATIONS] [REGULARIZATION LAMBDA] [TEST_DATA; optional] [DEBUG; optional]" << std::endl;
    return 1;
  }

  std::cout << "Train: " << argv[1] << std::endl;
  std::cout << "Learning rate: " << argv[2] << std::endl;
  std::cout << "Max iterations: " << argv[3] << std::endl;
  std::cout << "Regularization lambda: " << argv[4] << std::endl;

  if (argc >= 6)
  {
    std::cout << "Test: " << argv[5] << std::endl;
  }
    
	LogisticRegression logistic_reg;
	logistic_reg.LoadTrainDataFromDelimit(argv[1], ',');
  logistic_reg.NormalizeFeatures();
  logistic_reg.AddInterceptColumn();

  if (ConvertStringToDouble(argv[4]) == 0)
  {
    // Without regularization
    logistic_reg.GradientDescent(ConvertStringToDouble(argv[2]), ConvertStringToDouble(argv[3]));
  }
  else
  {
    // With regularization
    logistic_reg.GradientDescentWithRegularization(ConvertStringToDouble(argv[2]), ConvertStringToDouble(argv[3]), ConvertStringToDouble(argv[4]));
  }

  if (argc >= 6)
  {
    logistic_reg.LoadTestDataFromDelimit(argv[5], ',');
    
    // Evaluate predictions
    //std::vector<double> predictions = logistic_reg.Predict();
    //logistic_reg.EvaluatePredictions(predictions);

    // Evaluate classification labels
    std::vector<double> labels = logistic_reg.BinaryClassify();
    logistic_reg.EvaluateClassificationLabels(labels);
  }

  if (argc == 7)
  {
    // Debug mode is on; print more information
    logistic_reg.PrintDebugInfo();
  }

	return 0;
}

