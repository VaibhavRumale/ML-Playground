#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

class StandardAttention {
public:
    StandardAttention(int d_model) : scale(std::sqrt(d_model)) {
        query = MatrixXd::Random(d_model, d_model);
        key = MatrixXd::Random(d_model, d_model);
        value = MatrixXd::Random(d_model, d_model);
    }

    MatrixXd forward(const MatrixXd& x) {
        MatrixXd Q = x * query;
        MatrixXd K = x * key;
        MatrixXd V = x * value;

        MatrixXd attn_scores = Q * K.transpose() / scale;

        #pragma omp parallel for
        for (int i = 0; i < attn_scores.rows(); ++i) {
            attn_scores.row(i) = softmax(attn_scores.row(i));
        }

        MatrixXd output = attn_scores * V;
        return output;
    }

private:
    MatrixXd query, key, value;
    double scale;

    VectorXd softmax(const VectorXd& x) {
        VectorXd exp_x = x.array().exp();
        return exp_x / exp_x.sum();
    }
};

class FlashAttention {
public:
    FlashAttention(int d_model) : scale(std::sqrt(d_model)) {
        query = MatrixXd::Random(d_model, d_model);
        key = MatrixXd::Random(d_model, d_model);
        value = MatrixXd::Random(d_model, d_model);
    }

    MatrixXd forward(const MatrixXd& x) {
        MatrixXd Q = x * query;
        MatrixXd K = x * key;
        MatrixXd V = x * value;

        MatrixXd output = MatrixXd::Zero(Q.rows(), V.cols());

        #pragma omp parallel for
        for (int i = 0; i < Q.rows(); ++i) {
            VectorXd q_i = Q.row(i);
            VectorXd softmax_sum = VectorXd::Zero(K.rows());

            for (int j = 0; j < K.rows(); ++j) {
                double score = q_i.dot(K.row(j)) / scale;
                softmax_sum[j] = std::exp(score);
            }

            softmax_sum /= softmax_sum.sum();

            for (int j = 0; j < K.rows(); ++j) {
                output.row(i) += softmax_sum[j] * V.row(j);
            }
        }

        return output;
    }

private:
    MatrixXd query, key, value;
    double scale;
};

void benchmark_attention(StandardAttention& std_attn, FlashAttention& flash_attn, const MatrixXd& input) {
    auto start = high_resolution_clock::now();
    MatrixXd output_std = std_attn.forward(input);
    auto stop = high_resolution_clock::now();
    auto duration_std = duration_cast<microseconds>(stop - start);
    cout << "Standard Attention Time (OpenMP): " << duration_std.count() << " microseconds" << endl;

    start = high_resolution_clock::now();
    MatrixXd output_flash = flash_attn.forward(input);
    stop = high_resolution_clock::now();
    auto duration_flash = duration_cast<microseconds>(stop - start);
    cout << "Flash Attention Time (OpenMP): " << duration_flash.count() << " microseconds" << endl;
}

int main() {
    int d_model = 64;
    int seq_length = 128;

    MatrixXd input = MatrixXd::Random(seq_length, d_model);

    StandardAttention standard_attention(d_model);
    FlashAttention flash_attention(d_model);

    benchmark_attention(standard_attention, flash_attention, input);

    return 0;
}

