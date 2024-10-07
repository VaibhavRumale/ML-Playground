#include <iostream>
#include <vector>
#include <cmath>

double mean(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val;
    }
    return sum / v.size();
}

double covariance(const std::vector<double>& x, const std::vector<double>& y, double mean_x, double mean_y) {
    double cov = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return cov;
}

double variance(const std::vector<double>& x, double mean_x) {
    double var = 0.0;
    for (double val : x) {
        var += std::pow(val - mean_x, 2);
    }
    return var;
}

std::pair<double, double> coefficients(const std::vector<double>& x, const std::vector<double>& y) {
    double mean_x = mean(x);
    double mean_y = mean(y);

    double b1 = covariance(x, y, mean_x, mean_y) / variance(x, mean_x);
    double b0 = mean_y - b1 * mean_x;

    return {b0, b1};
}

std::vector<double> predict(const std::vector<double>& x, double b0, double b1) {
    std::vector<double> predictions;
    for (double val : x) {
        predictions.push_back(b0 + b1 * val);
    }
    return predictions;
}

int main() {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.2, 1.9, 3.2, 4.1, 5.3};

    auto [b0, b1] = coefficients(x, y);

    std::cout << "Intercept (b0): " << b0 << std::endl;
    std::cout << "Slope (b1): " << b1 << std::endl;

    std::vector<double> predictions = predict(x, b0, b1);

    std::cout << "Predictions: ";
    for (double val : predictions) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

