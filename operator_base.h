/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#ifndef TINYDNN__OPERATOR_BASE_H
#define TINYDNN__OPERATOR_BASE_H
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Geometry>

class Variable;
class GraphManager;

class OperatorBase {
    /*Operator是计算图的边 前馈计算输入输出是Variable 可以多次前馈计算 每次前馈计算会输出一个新Variable
     * 记录自身的参数 记录输入输出pair 反馈计算读取输出Variable的梯度 并计算赋值更新输入Variable的梯度*/
public:
    std::string _name;
    std::unordered_map<std::string, Eigen::MatrixXd> _gradientOfOutput;
    std::vector<std::pair<std::string, std::string>> _inputOutputPair;
    /*图管理器 记录全体operator与variable的信息*/
    GraphManager &_graphManager;
    explicit OperatorBase(GraphManager &);
    virtual ~OperatorBase();
    /*每个operator还有自身参数 自身参数的梯度 前馈反馈的逻辑 由派生类实现*/
    virtual Variable forward(Variable &v);
    virtual Variable forward(Variable &v1,
                             Variable &v2);
    virtual void backward();
    /*更新参数 载入参数 保存参数 初始化参数 如果layer没有参数可以不操作*/
    virtual void update();
    virtual bool loadWeight();
    virtual bool saveWeight();
    virtual void initWeight(uint64_t seed);
};

#endif //TINYDNN__OPERATOR_BASE_H
