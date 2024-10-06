#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

class LogisticRegression {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int num_iterations;

public:
    LogisticRegression(double lr = 0.01, int iters = 1000) 
        : learning_rate(lr), num_iterations(iters), bias(0.0) {}

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();

        weights.resize(n_features, 0.0);

        for (int iter = 0; iter < num_iterations; ++iter) {
            std::vector<double> y_pred(n_samples, 0.0);

            for (int i = 0; i < n_samples; ++i) {
                double linear_model = bias;
                for (int j = 0; j < n_features; ++j) {
                    linear_model += weights[j] * X[i][j];
                }
                y_pred[i] = sigmoid(linear_model);
            }

            std::vector<double> dw(n_features, 0.0);
            double db = 0.0;

            for (int i = 0; i < n_samples; ++i) {
                double error = y_pred[i] - y[i];
                for (int j = 0; j < n_features; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            for (int j = 0; j < n_features; ++j) {
                weights[j] -= learning_rate * dw[j] / n_samples;
            }
            bias -= learning_rate * db / n_samples;
        }
    }

    int predict(const std::vector<double>& x) {
        double linear_model = bias;
        for (int j = 0; j < x.size(); ++j) {
            linear_model += weights[j] * x[j];
        }
        return sigmoid(linear_model) >= 0.5 ? 1 : 0;
    }

    double predict_probability(const std::vector<double>& x) {
        double linear_model = bias;
        for (int j = 0; j < x.size(); ++j) {
            linear_model += weights[j] * x[j];
        }
        return sigmoid(linear_model);
    }

    void print_parameters() {
        std::cout << "Weights: ";
        for (double w : weights) {
            std::cout << std::fixed << std::setprecision(4) << w << " ";
        }
        std::cout << "\nBias: " << std::fixed << std::setprecision(4) << bias << "\n";
    }
};

int main() {
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, 
        {2.0, 3.0}, 
        {3.0, 4.0}, 
        {5.0, 6.0}
    };
    std::vector<int> y = {0, 0, 1, 1};

    LogisticRegression model(0.1, 1000);

    model.train(X, y);

    model.print_parameters();

    std::vector<double> new_sample = {4.0, 5.0};
    int prediction = model.predict(new_sample);
    double probability = model.predict_probability(new_sample);

    std::cout << "Predicted class: " << prediction << "\n";
    std::cout << "Predicted probability: " << probability << "\n";

    return 0;
}

