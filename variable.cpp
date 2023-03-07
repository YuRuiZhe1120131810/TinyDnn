/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include <utility>
#include <iostream>
#include "variable.h"
#include "graph_manager.h"
class GraphManager;

Variable::Variable(const Eigen::MatrixXd &m,
                   std::string name) : _name(std::move(name)) {
    _gradientOfLoss.resize(0,
                           0);
    _gradientOfOperator.clear();
    _value = m;
    GraphManager::get()._variables.emplace(_name,
                                           std::shared_ptr<Variable>(this));
}

Variable::Variable(Variable &&other) noexcept {
    std::swap(_value,
              other._value);
    std::swap(_gradientOfLoss,
              other._gradientOfLoss);
    std::swap(_gradientOfOperator,
              other._gradientOfOperator);
    /*GraphManager 会把 Variable 的地址记录一份 移动构造后右值引用变量自动释放 防止指针悬空需要重新记录 Variable 地址*/
    GraphManager::get()._variables.at(other._name) = std::shared_ptr<Variable>(this);
    other.~Variable();
}

Variable &Variable::operator=(Variable &&other) noexcept {
    if (this != &other) {
        std::swap(_value,
                  other._value);
        std::swap(_gradientOfLoss,
                  other._gradientOfLoss);
        std::swap(_gradientOfOperator,
                  other._gradientOfOperator);
        /*GraphManager 会把 Variable 的地址记录一份 当移动赋值后右值引用变量自动释放 防止指针悬空需要重新记录 Variable 地址*/
        GraphManager::get()._variables.at(other._name) = std::shared_ptr<Variable>(this);
        other.~Variable();
    }
    return *this;
}

Variable::~Variable() {
    _value.resize(0,
                  0);
    _gradientOfLoss.resize(0,
                           0);
    for (auto &pair:_gradientOfOperator) {
        pair.second.resize(0,
                           0);
    }
    _gradientOfOperator.clear();
}

uint32_t Variable::rows() const {
    return _value.rows();
}

uint32_t Variable::cols() const {
    return _value.cols();
}

Eigen::MatrixXd Variable::val() const {
    return _value;
}

