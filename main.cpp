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
#include "utility.h"
#include <random>

int main(int argc,
         char **argv) {
    /*模拟训练数据准备*/
    const int data_cnt_ = 4, data_dim_ = 2;
    std::vector<double> pts_val_{.5, 0, .5, 1, 1, .5, 0, .5};
    std::vector<double> label_val_{1, 0, 1, 0, 0, 1, 0, 1};
    Eigen::MatrixXd pts_ = Eigen::MatrixXd::Map(&pts_val_[0],
                                                data_cnt_,
                                                data_dim_);
    Eigen::MatrixXd label_ = Eigen::MatrixXd::Map(&label_val_[0],
                                                  data_cnt_,
                                                  data_dim_);
    std::cout << "预估数据=" << std::endl << pts_ << std::endl << "预估标签=" << std::endl << label_ << std::endl;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0,
                                                data_cnt_);
    /*网络层定义*/
    FullConnect layer_0_(data_dim_,
                         4,
                         1e-4,
                         std::string("relu"),
                         GraphManager::get());
    FullConnect layer_2_(4,
                         data_dim_,
                         1e-4,
                         std::string("softmax"),
                         GraphManager::get());
    CrossEntropyLoss layer_loss_(GraphManager::get());
    for (size_t cnt_{0}; cnt_ < 10000; ++cnt_) {
        /*随机交换一行做数据增强*/
        const int row_a_{static_cast<int>(dist(mt)) % data_cnt_}, row_b_{static_cast<int>(dist(mt)) % data_cnt_};
        pts_.row(row_a_).swap(pts_.row(row_b_));
        label_.row(row_a_).swap(label_.row(row_b_));
        /*训练样本*/
        Variable pts_var_(pts_,
                          std::string("train_data"),
                          GraphManager::get());
        /*拟合目标*/
        Variable label_var_(label_,
                            std::string("train_label"),
                            GraphManager::get());
        /*前馈*/
        Variable x_ = layer_0_.forward(pts_var_);
        Variable z_ = layer_2_.forward(x_);
        Variable loss_ = layer_loss_.forward(z_,
                                             label_var_);
        const double current_loss_ = loss_._value.coeff(0,
                                                        0);
        Eigen::MatrixXd fit_success_ = Utility::matSame(Utility::oneHot(z_._value),
                                                        label_);
        std::cout << "正确拟合的样本数量=" << fit_success_.sum() << ",迭代轮次=" << cnt_ << ",当前loss=" << current_loss_
                  << std::endl;
        if (static_cast<int>(fit_success_.sum()) == data_cnt_) {
            break;
        }
        /*反馈*/
        GraphManager::get().backward(loss_);
        /*用梯度更新参数*/
        GraphManager::get().update();
        /*释放临时Variable*/
        GraphManager::get().release();
    }

    std::cout << "main() exit" << std::endl;
    return 0;
}
