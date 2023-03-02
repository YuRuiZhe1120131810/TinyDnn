/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include <iostream>
#include "graph_manager.h"
#include "variable.h"
#include "full_connect.h"
#include "cross_entropy_loss.h"

int main(int argc,
         char **argv) {
    /*模拟训练数据准备*/
    const int data_cnt_ = 4, data_dim_ = 2;
    std::vector<double> pts_val_{0., 1., 0., 1., 0., 1., 1., 0.};
    std::vector<double> label_val_{1., 1., 0., 0., 0., 0., 1., 1.};
    Eigen::MatrixXd pts_ = Eigen::MatrixXd::Map(&pts_val_[0],
                                                data_cnt_,
                                                data_dim_);
    Eigen::MatrixXd label_ = Eigen::MatrixXd::Map(&label_val_[0],
                                                  data_cnt_,
                                                  data_dim_);
//    std::cout << "label:\n" << label_ << std::endl << "pts:\n" << pts_ << std::endl;
    Variable pts_var_(pts_,
                      std::string("train_data"),
                      GraphManager::get());
    Variable label_var_(label_,
                        std::string("train_label"),
                        GraphManager::get());
    /*网络层定义*/
    FullConnect layer_0_(data_dim_,
                         3,
                         1e-5,
                         std::string(""),
                         GraphManager::get());
    FullConnect layer_1_(3,
                         data_dim_,
                         1e-5,
                         std::string("softmax"),
                         GraphManager::get());
    CrossEntropyLoss layer_2_(GraphManager::get());
    /*前馈*/
    Variable x = layer_0_.forward(pts_var_);
    Variable y = layer_1_.forward(x);
    Variable loss_ = layer_2_.forward(y,
                                      label_var_);
    /*反馈*/
    GraphManager::get().backward(loss_);
    /*用梯度更新参数*/
    GraphManager::get().update();
    std::cout << "main() exit" << std::endl;
    return 0;
}
