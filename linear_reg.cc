#include "linear_reg.h"
#include "utilz/utilz.h"

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

// Load the training data into a matrix from a delimit-separated format
// Params: Path to the training data in a delimit-separated format
// Return: 1 if something goes wrong
int LinearRegression::LoadTrainDataFromDelimit(const String path, const char delimit)
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
int LinearRegression::LoadTestDataFromDelimit(const String path, const char delimit)
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
void LinearRegression::NormalizeFeatures()
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

// Add one column of 1s to the train_data matrix (as the intercept)
void LinearRegression::AddInterceptColumn(boost::numeric::ublas::matrix<double> & data)
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

// Gradient descent to find the best fit
// Params: the learning rate alpha, and the number of iterations
// Return: the vector holding the costs through the iterations
std::vector<double> LinearRegression::GradientDescent(const float alpha, const int max_iteration)
{
  namespace bnu = boost::numeric::ublas;

  AddInterceptColumn(train_data);

  theta = bnu::zero_matrix<double> (train_data.size2(), 1);   // Initialize the theta matrix
  vector<double> costs_history;     // Initialize the vector to hold the costs through the iterations

  bnu::matrix<double> diff; 

  int iteration = 0;

  std::cout << "Starting gradient descent..." << std::endl;

  while (iteration < max_iteration) 
  {
    iteration++;

    diff = bnu::prod(bnu::trans(train_data), (bnu::prod(train_data, theta) - target));
    theta -= diff * alpha / train_data.size1();
    
    double costs = ComputeCosts();
    
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
double LinearRegression::ComputeCosts()
{
  namespace bnu = boost::numeric::ublas;

  double costs;
  bnu::matrix<double> term = bnu::prod(train_data, theta) - target;
  bnu::matrix<double> diff = bnu::element_prod(term, term) / (2 * train_data.size1());

  for (int i = 0; i < train_data.size1(); i++)
  {
    costs += diff(i, 0);
  }

  return costs;
}

// Predict a single case using the theta learned from gradient descent
// Params: the single point data for the prediction
// Return: the predicted value
double LinearRegression::Predict(const std::vector<double> data)
{
  double prediction = theta(0, 0) * data[0];    // Initialize with the intercept

  for (int i = 1; i < data.size(); i++)
  {
    prediction += theta(i, 0) * ((data[i] - train_data_mean(0, i - 1)) / train_data_stddev(0, i - 1)); 
  }

  return prediction;
}

// Predict multiple cases using the theta learned from gradient descent
// Params: the data for the prediction
// Return: vector of the predicted value
std::vector<double> LinearRegression::Predict(boost::numeric::ublas::matrix<double> data)
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

    predictions.push_back(predict);
  }

  return predictions;
}

// Predict the test_data using the theta learned from gradient descent
// Params: N/A
// Return: vector of the predicted value
std::vector<double> LinearRegression::Predict()
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

    predictions.push_back(predict);
  }

  return predictions;
}

// Evaluate the predictions with the test_target
void LinearRegression::EvaluatePredictions(std::vector<double> predictions)
{
  std::cout << "Predict" << "\t\t" << "True" << "\t\t" << "Diff (in perc.)" << std::endl;
  for (int i = 0; i < predictions.size(); i++)
  {
    std::cout << predictions[i] << "\t\t" << test_target(i, 0) << "\t\t" << (abs(predictions[i] - test_target(i,0)) / test_target(i, 0) * 100) << "%" << std::endl;
  }

}

// Print more information when the debug mode is ON
void LinearRegression::PrintDebugInfo()
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

LinearRegression::LinearRegression()
{
  // Threshold for the convergence of the gradient descent process
  epsilon = 1e-8;
}
	

LinearRegression::~LinearRegression()
{

}



int main (int argc, char* argv[]) 
{
  if (argc < 4) 
  { 
    // We expect at least 4 arguments: 
    // 1. Path to the data for training (required)
    // 2. The learning rate (required)
    // 3. The maximum number of iteration (required)
    // 4. Path to the data for testing (optional)
    // 5. Toggle debug mode (optional)
    std::cerr << "Usage: " << argv[0] << " [TRAIN_DATA] [LEARNING_RATE] [MAX_ITERATIONS] [TEST_DATA; optional] [DEBUG; optional]" << std::endl;
    return 1;
  }

  std::cout << "Train: " << argv[1] << std::endl;
  std::cout << "Learning rate: " << argv[2] << std::endl;
  std::cout << "Max iterations: " << argv[3] << std::endl;

  if (argc >= 5)
  {
    std::cout << "Test: " << argv[4] << std::endl;
  }
    
	LinearRegression linear_reg;
	linear_reg.LoadTrainDataFromDelimit(argv[1], ',');
  linear_reg.NormalizeFeatures();
  linear_reg.GradientDescent(ConvertStringToDouble(argv[2]), ConvertStringToDouble(argv[3]));

  if (argc >= 5)
  {
    linear_reg.LoadTestDataFromDelimit(argv[4], ',');

    std::vector<double> predictions = linear_reg.Predict();

    linear_reg.EvaluatePredictions(predictions);
  }

  if (argc == 6)
  {
    // Debug mode is on; print more information
    linear_reg.PrintDebugInfo();
  }

	return 0;
}

