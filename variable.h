/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#ifndef EIGEN__VARIABLE_H
#define EIGEN__VARIABLE_H
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Geometry>

class GraphManager;
class Variable {
    /*Variable是计算图的节点 可以被多个Operator调用 记录自身的值 最终loss对自身的梯度 调用Operator对自身的梯度
     * 汇集各调用Operator对自身的梯度后进行加和 作为自身最终的梯度*/
public:
    std::string _name;
    std::unordered_map<std::string, Eigen::MatrixXd> _gradientOfOperator;
    Eigen::MatrixXd _gradientOfLoss;
    Eigen::MatrixXd _value;
    /*禁用拷贝构造，根据Variable构造另一个Variable是无意义的，构造计算图节点不应当依赖另一个节点*/
    Variable(const Variable &other) = delete;
    explicit Variable(const Eigen::MatrixXd &m,
                      std::string name,
                      GraphManager &graph_manager);
    /*禁用拷贝赋值，拿Variable给已初始化的Variable赋值是无意义的，计算图节点不能覆盖另一个节点*/
    Variable &operator=(const Variable &other) = delete;
    /*移动赋值，用于临时变量保存结果*/
    Variable &operator=(Variable &&other) noexcept;
    ~Variable() = default;
    void reset();
    uint rows() const;
    uint cols() const;
    Eigen::MatrixXd val() const;
};
#endif //EIGEN__VARIABLE_H
