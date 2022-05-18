/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include <utility>
#include "variable.h"
#include "graph_manager.h"
class GraphManager;

Variable::Variable(const Eigen::MatrixXd &m,
                   std::string name,
                   GraphManager &graph_manager) : _name(std::move(name)) {
    reset();
    _value = m;
    graph_manager._variables.emplace(_name,
                                     std::shared_ptr<Variable>(this));
}

Variable::Variable(Variable &&other) noexcept {
    std::swap(_value,
              other._value);
    std::swap(_gradientOfLoss,
              other._gradientOfLoss);
    std::swap(_gradientOfOperator,
              other._gradientOfOperator);
    other.reset();
}

void Variable::reset() {
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
        other.reset();
    }
    return *this;
}

uint Variable::rows() const {
    return _value.rows();
}

uint Variable::cols() const {
    return _value.cols();
}

Eigen::MatrixXd Variable::val() const {
    return _value;
}

