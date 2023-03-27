/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include <iostream>
#include <cmath>
#include <utility>
#include <numeric>
#include "graph_manager.h"
#include "variable.h"
#include "full_connect.h"
#include "cross_entropy_loss.h"

int main(int argc,
         char **argv) {
    srand(4);
    /*模拟训练数据准备*/
    const int data_cnt_ = 4, data_dim_ = 2;
    std::vector<double> pts_val_{2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0};
    std::vector<double> label_val_{1, 1, 0, 0, 0, 0, 1, 1};
    Eigen::MatrixXd pts_ = Eigen::MatrixXd::Map(&pts_val_[0],
                                                data_cnt_,
                                                data_dim_);
    Eigen::MatrixXd label_ = Eigen::MatrixXd::Map(&label_val_[0],
                                                  data_cnt_,
                                                  data_dim_);
    /*把预估结果转为ont_hot编码*/
    auto one_hot_ = [](const Eigen::MatrixXd &m,
                       const bool row_size = true)->Eigen::MatrixXd {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m.rows(),
                                                       m.cols());
        Eigen::MatrixXf::Index idx_;
        for (int r = 0; row_size && r < m.rows(); ++r) {
            m.row(r).maxCoeff(&idx_);
            result.coeffRef(r,
                            idx_) = 1;
        }
        for (int c = 0; !row_size && c < m.cols(); ++c) {
            m.col(c).maxCoeff(&idx_);
            result.coeffRef(idx_,
                            c) = 1;
        }
        return result;
    };
    /*统计训练样本是否拟合完成*/
    std::bitset<data_cnt_> fit_success_;
    /*网络层定义*/
    FullConnect layer_0_(data_dim_,
                         3,
                         1e-4,
                         std::string("relu"),
                         GraphManager::get());
    FullConnect layer_1_(3,
                         data_dim_,
                         1e-4,
                         std::string("softmax"),
                         GraphManager::get());
    CrossEntropyLoss layer_2_(GraphManager::get());
    for (struct { double _oprimalLoss{INFINITY};bool _runFlag{true}; uint64_t _cnt{0}; } condition_;
         condition_._runFlag; ++condition_._cnt) {
        /*随机采样一个训练样本和拟合目标*/
        const Eigen::MatrixXd pts_single_ = pts_.block(condition_._cnt % data_cnt_,
                                                       0,
                                                       1,
                                                       data_dim_);
        const Eigen::MatrixXd label_single_ = label_.block(condition_._cnt % data_cnt_,
                                                           0,
                                                           1,
                                                           data_dim_);
        /*训练样本*/
        Variable pts_var_(pts_single_,
                          std::string("train_data"),
                          GraphManager::get());
        /*拟合目标*/
        Variable label_var_(label_single_,
                            std::string("train_label"),
                            GraphManager::get());
        /*前馈*/
        Variable x = layer_0_.forward(pts_var_);
        Variable y = layer_1_.forward(x);
        Variable loss_ = layer_2_.forward(y,
                                          label_var_);
        const double current_loss_ = loss_._value.coeff(0,
                                                        0);
        fit_success_ = one_hot_(y._value).isApprox(label_single_)
                       ? fit_success_.set(condition_._cnt % data_cnt_) : fit_success_;
        condition_._runFlag = fit_success_.count() != data_cnt_;
        condition_._oprimalLoss = std::min(current_loss_,
                                           condition_._oprimalLoss);
        /*反馈*/
        GraphManager::get().backward(loss_);
        /*用梯度更新参数*/
        GraphManager::get().update();
        /*打印初始结果 和 最优结果*/
        std::cout << "iter=" << condition_._cnt
                  //<< ",current_loss_=" << current_loss_ << ",optimal_loss=" << condition_._oprimalLoss
                  << ",prediction=" << y._value << ",label=" << label_single_
                  << ",success train samples=" << fit_success_.count() << std::endl;
        /*释放临时Variable*/
        GraphManager::get().release();
    }

    std::cout << "main() exit" << std::endl;
    return 0;
}
