/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include "graph_manager.h"
#include "variable.h"
#include <iostream>

int main(int argc,
         char **argv) {
    /*模拟训练数据准备*/
    const int data_cnt_ = 4;
    std::vector<double> pts_val_{0., 1., 0., 1., 0., 1., 1., 0.};
    std::vector<double> label_val_{1., 1., 0., 0.};
    const Eigen::MatrixXd pts_ = Eigen::MatrixXd::Map(&pts_val_[0],
                                                      data_cnt_,
                                                      2);
    const Eigen::MatrixXd label_ = Eigen::MatrixXd::Map(&label_val_[0],
                                                        data_cnt_,
                                                        1);
    std::cout << pts_ << std::endl;
    std::cout << label_ << std::endl;
}
