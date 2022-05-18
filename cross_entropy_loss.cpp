/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 普通常量名 大写字母开头 下划线结尾
*/
#include "cross_entropy_loss.h"
#include "operator_base.h"
#include "variable.h"
#include "graph_manager.h"
class OperatorBase;
class Variable;
class GraphManager;

CrossEntropyLoss::CrossEntropyLoss(GraphManager &graph_manager) : _forwardCount(0),
                                                                  OperatorBase(graph_manager) {
    _name = std::string("FullConnect_").append(std::to_string(_instanceCount++));/*构造实例计数增一*/
    _graphManager._operators.emplace(_name,
                                     std::shared_ptr<OperatorBase>(this));
}

Variable CrossEntropyLoss::forward(Variable &prediction,
                                   Variable &label) {
    assert(prediction.cols() == label.cols() && prediction.rows() == label.rows());
    assert(0 < prediction._value.minCoeff() && prediction._value.maxCoeff() <= 1 && 0 <= label._value.minCoeff()
               && label._value.maxCoeff() <= 1);
    /*计算loss=sum[-label*ln(prediction)]*/
    const Eigen::MatrixXd tmp_ = -label._value.array() * prediction._value.array().log();
    const Eigen::MatrixXd a_ = Eigen::MatrixXd::Ones(1,
                                                     tmp_.rows());
    const Eigen::MatrixXd b_ = Eigen::MatrixXd::Ones(tmp_.cols(),
                                                     1);
    auto output_ = Variable(a_ * tmp_ * b_,
                            _name + "_" + std::to_string(_forwardCount++),
                            _graphManager);
    /*填充节点连接关系*/
    _graphManager._variableCreateBy.emplace(output_._name,
                                            _name);
    _graphManager._variableCallBy[prediction._name].emplace_back(_name);
    _graphManager._variableCallBy[label._name].emplace_back(_name);
    _inputOutputPair.emplace_back(std::make_pair(prediction._name,
                                                 output_._name));
    _inputOutputPair.emplace_back(std::make_pair(label._name,
                                                 output_._name));
    auto as_diag_ = [](Eigen::MatrixXd x)->Eigen::MatrixXd {
        return Eigen::VectorXd::Map(x.data(),
                                    x.size()).asDiagonal().toDenseMatrix();
    };;
    /*输出loss对prediction的梯度*/
    Eigen::MatrixXd grad_prediction_ = as_diag_(prediction._value.cwiseInverse().eval())
        * label._value.asDiagonal().toDenseMatrix()
        * kroneckerProduct(b_.transpose(),
                           a_).transpose().eval();
    auto has_ = prediction._gradientOfOperator.find(_name);
    if (has_ == prediction._gradientOfOperator.end()) {
        prediction._gradientOfOperator.emplace(_name,
                                               grad_prediction_);
    }
    else {
        has_->second += grad_prediction_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
    }
    /*输出loss对label的梯度*/
    Eigen::MatrixXd grad_label_ = as_diag_(prediction._value.array().log().eval())
        * kroneckerProduct(b_.transpose(),
                           a_).transpose().eval();
    has_ = label._gradientOfOperator.find(_name);
    if (has_ == label._gradientOfOperator.end()) {
        label._gradientOfOperator.emplace(_name,
                                          grad_label_);
    }
    else {
        has_->second += grad_label_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
    }
    /*CrossEntropy没有参数 所以不需要更新_gradientOfOutput 只需要更新两个variable的_gradientOfOperator*/
    return output_;
}

void CrossEntropyLoss::backward() {
    for (const auto &ele :_inputOutputPair) {
        const auto &output_name_{ele.second};
        /*loss对z的梯度*/
        const auto &grad_loss_to_output_{_graphManager._variables[output_name_]->_gradientOfLoss};
        /*loss对x的梯度 = z对x的梯度 * loss对z的梯度*/
        const auto &input_name_{ele.first};
        const Eigen::MatrixXd tmp_ = _graphManager._variables[input_name_]->_gradientOfOperator[_name]
            * grad_loss_to_output_;
        auto &grad_input_ = _graphManager._variables[input_name_]->_gradientOfLoss;
        if (grad_input_.size() == 0) {
            grad_input_ = tmp_;
        }
        else {
            grad_input_ += tmp_;
        }
    }
}
