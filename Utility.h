/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#ifndef TINYDNN__UTILITY_H
#define TINYDNN__UTILITY_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class Utility {
public:
    /*把预估结果转为ont_hot编码*/
    static Eigen::MatrixXd oneHot(const Eigen::MatrixXd &,
                                  bool row_wise = true);
    /*逐行或逐列对比两个矩阵是否相同 逐行对比时 返回一个单列矩阵 逐列对比时 返回一个单行矩阵*/
    static Eigen::MatrixXd matSame(const Eigen::MatrixXd &a,
                                   const Eigen::MatrixXd &b,
                                   bool row_wise = true);
};

#endif //TINYDNN__UTILITY_H
