/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#ifndef TINYDNN__FULL_CONNECT_H
#define TINYDNN__FULL_CONNECT_H
#include <memory>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Geometry>
#include <iostream>
#include "operator_base.h"
class OperatorBase;
class Variable;
class GraphManager;

class FullConnect : public OperatorBase {
public:
    static uint32_t _instanceCount;/*记录有多少个FullConnect对象*/
    uint32_t _forwardCount;/*记录forward次数*/
    Eigen::MatrixXd _weight;/*行=输入维 列=输出维*/
    Eigen::MatrixXd _bias;/*行=1 列=输出维*/
    std::unordered_map<std::string, Eigen::MatrixXd> _wGradOfOutput;/*每次前馈记录梯度*/
    std::unordered_map<std::string, Eigen::MatrixXd> _bGradOfOutput;/*每次前馈记录梯度*/
    Eigen::MatrixXd _gradWeight;/*反馈汇总梯度*/
    Eigen::MatrixXd _gradBias;/*反馈汇总梯度*/
    double _learning_rate;/*学习率*/
    std::string _act_func;/*激活函数*/
    explicit FullConnect(uint64_t in_channel,
                         uint64_t out_channel,
                         double learning_rate,
                         std::string act_func,
                         GraphManager &graph_manager);
    Variable forward(Variable &input) override;
    void backward() override;
    void update() override;
    void clearGrad() override;
};

#endif //TINYDNN__FULL_CONNECT_H
