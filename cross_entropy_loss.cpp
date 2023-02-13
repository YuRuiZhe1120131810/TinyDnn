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

uint32_t CrossEntropyLoss::_instanceCount{0};

CrossEntropyLoss::CrossEntropyLoss(GraphManager &graph_manager) : _forwardCount(0),
                                                                  OperatorBase(graph_manager) {
    _name = std::string("CrossEntropy_").append(std::to_string(_instanceCount++));/*构造实例计数增一*/
    _graphManager._operators.emplace(_name,
                                     std::shared_ptr<OperatorBase>(this));
}

Variable CrossEntropyLoss::forward(Variable &prediction,
                                   Variable &label) {
    assert(prediction.cols() == label.cols() && prediction.rows() == label.rows());
    assert(0 < prediction._value.minCoeff() && prediction._value.maxCoeff() <= 1 && 0 <= label._value.minCoeff()
               && label._value.maxCoeff() <= 1);
    /*计算loss=sum[-label*ln(prediction)]*/
    const double result_ = (-label._value.array() * prediction._value.array().log()).sum();
    auto output_ = Variable(Eigen::Matrix<double, 1, 1>(result_),
                            _name + "_" + std::to_string(_forwardCount++),
                            _graphManager);
    output_._gradientOfLoss = Eigen::Matrix<double, 1, 1>(1.0);
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
    };
    /* CrossEntropy的输出就是loss，loss对CE_layer的梯度始终设置为单位矩阵
     * 输出loss对prediction的梯度 = CrossEntropy对prediction的梯度 = CE_layer对prediction的梯度 * loss对CrossEntropy的梯度
     * 输出loss对label的梯度 = CrossEntropy对label的梯度 = CrossEntropy对label的梯度 * loss对CrossEntropy的梯度*/
    Eigen::MatrixXd prediction_grad_ = -label._value.array() * prediction._value.cwiseInverse().array();
    prediction._gradientOfOperator.emplace(_name,
                                           prediction_grad_.reshaped());
    Eigen::MatrixXd label_grad_ = -prediction._value.array().log();
    label._gradientOfOperator.emplace(_name,
                                      label_grad_.reshaped());
    return output_;
}

void CrossEntropyLoss::backward() {
    for (const auto &ele :_inputOutputPair) {
        /*loss对x的梯度 = z对x的梯度 * loss对z的梯度*/
        const std::string &input_name_{ele.first}, &output_name_{ele.second};
        Variable &layer_input_ = *_graphManager._variables.at(input_name_);
        Variable &layer_output_ = *_graphManager._variables.at(output_name_);
        const auto &grad_loss_to_output_{layer_output_._gradientOfLoss};
        const Eigen::MatrixXd tmp_ = layer_input_._gradientOfOperator.at(_name) * grad_loss_to_output_;
        if (layer_input_._gradientOfLoss.size() == 0) {
            layer_input_._gradientOfLoss = tmp_;
        }
        else {
            layer_input_._gradientOfLoss += tmp_;
        }
    }
}
