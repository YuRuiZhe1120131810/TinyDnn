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
                   std::string name,
                   GraphManager &graph_manager) : _name(std::move(name)) {
    _value = m;
    _gradientOfLoss.resize(0,
                           0);
    _gradientOfOperator.clear();
    graph_manager._variables.emplace(_name,
                                     this);
}

Variable::Variable(Variable &&other) noexcept {
    std::swap(_value,
              other._value);
    std::swap(_gradientOfLoss,
              other._gradientOfLoss);
    std::swap(_gradientOfOperator,
              other._gradientOfOperator);
    other.~Variable();
}

Variable::~Variable() {
    _value.resize(0,
                  0);
    _gradientOfLoss.resize(0,
                           0);
    _gradientOfOperator.clear();
}

Variable &Variable::operator=(Variable &&other) noexcept {
    if (this != &other) {
        std::swap(_value,
                  other._value);
        std::swap(_gradientOfLoss,
                  other._gradientOfLoss);
        std::swap(_gradientOfOperator,
                  other._gradientOfOperator);
        other.~Variable();
    }
    return *this;
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

