/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 每个Variable扇出为1 只能作为一个LAYER的输入 不能作为多个LAYER的输入
*/
#ifndef EIGEN__VARIABLE_H
#define EIGEN__VARIABLE_H
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Geometry>
class Variable {
    /*Variable是计算图的节点 可以被多个Operator调用 记录自身的值 自身的梯度 记录Operator对自身的梯度
     * 汇集各调用Operator对自身的梯度后进行加和 作为自身最终的梯度*/
    std::string _name;
    std::unordered_map<std::string, Eigen::MatrixXd> _gradient;
};
#endif //EIGEN__VARIABLE_H
