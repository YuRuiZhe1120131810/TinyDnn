/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 普通常量名 大写字母开头 下划线结尾
*/
#include <utility>
#include <algorithm>
#include "full_connect.h"
#include "operator_base.h"
#include "variable.h"
#include "graph_manager.h"
class OperatorBase;
class Variable;
class GraphManager;

FullConnect::FullConnect(uint64_t in_channel,
                         uint64_t out_channel,
                         double learning_rate,
                         std::string act_func,
                         GraphManager &graph_manager) : _act_func(std::move(act_func)),
                                                        _learning_rate(learning_rate),
                                                        _forwardCount(0),
                                                        _weightAndBias(Eigen::MatrixXd::Random(in_channel,
                                                                                               out_channel)),
                                                        OperatorBase(graph_manager) {
    _name = std::string("FullConnect_").append(std::to_string(_instanceCount++));/*构造实例计数增一*/

    _graphManager._operators.emplace(_name,
                                     std::shared_ptr<OperatorBase>(this));
}

Variable FullConnect::forward(Variable &input) {
    assert(input.cols() + 1 == _weightAndBias.rows() && "输入需要多一维方便拼接bias");
    /*扩大一个维度*/
    Eigen::MatrixXd input_expand_(input.rows(),
                                  input.cols() + 1);
    input_expand_ << input.val(), Eigen::MatrixXd::Ones(input.rows(),
                                                        1);
    /*计算y=xw+b*/
    auto output_ = Variable(input_expand_ * _weightAndBias,
                            _name + "_" + std::to_string(_forwardCount++),
                            _graphManager);
    /*填充节点连接关系*/
    _graphManager._variableCreateBy.emplace(output_._name,
                                            _name);
    _graphManager._variableCallBy[input._name].emplace_back(_name);
    _inputOutputPair.emplace_back(std::make_pair(input._name,
                                                 output_._name));
    /*y对w的梯度*/
    Eigen::MatrixXd identity_ = Eigen::MatrixXd::Identity(_weightAndBias.cols(),
                                                          _weightAndBias.cols());
    Eigen::MatrixXd grad_w_ = kroneckerProduct(identity_,
                                               input_expand_).transpose().eval();
    /*y对x的梯度*/
    identity_ = Eigen::MatrixXd::Identity(input.rows(),
                                          input.rows());
    Eigen::MatrixXd grad_input_ = kroneckerProduct(_weightAndBias.transpose(),
                                                   identity_).transpose().eval();
    auto sigmoid = [&](const Eigen::MatrixXd &x)->Eigen::MatrixXd { return 1 / ((-x).array().exp() + 1); };
    /*激活函数*/
    if (_act_func == "relu") {
        /*z=relu(y)*/
        output_._value = output_._value.cwiseMax(0);
        /*z对y的梯度*/
        Eigen::MatrixXd to_diag_ = Eigen::VectorXd::Map(output_._value.data(),
                                                        output_._value.size()).asDiagonal();
        Eigen::MatrixXd grad_relu_ = to_diag_.array().sign();
        /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
        _gradientOfOutput.emplace(output_._name,
                                  grad_w_ * grad_relu_);
        /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
        auto has_ = input._gradientOfOperator.find(_name);
        if (has_ == input._gradientOfOperator.end()) {
            input._gradientOfOperator.emplace(_name,
                                              grad_input_ * grad_relu_);
        }
        else {
            has_->second += grad_input_ * grad_relu_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
        }
    }
    else if (_act_func == "softmax") {
        /*softmax不是逐元素函数 要求输入输出都是向量*/
        /*z=softmax(y)*/
        Eigen::MatrixXd max_value_ = output_._value.rowwise().maxCoeff() * Eigen::MatrixXd::Ones(1,
                                                                                                 output_.cols());
        Eigen::MatrixXd output_val_ = (output_._value - max_value_).array().exp().matrix();/*防止浮点溢出*/
        output_val_.rowwise().normalize();/*均一化*/
        output_._value = output_val_;
        /*z对y的梯度 需要由多个分块对角矩阵拼接而成 z的每一行都计算一个梯度矩阵 按位置取梯度矩阵元素拼接成对角矩阵*/
        std::vector<Eigen::MatrixXd> blocks_;
        blocks_.reserve(output_.rows());
        for (const auto row : output_._value.rowwise()) {
            Eigen::MatrixXd tmp_ = -row.transpose() * row;
            tmp_.diagonal().setZero();
            const Eigen::MatrixXd diag_ = row.asDiagonal();
            tmp_ += diag_ - diag_ * diag_;
            blocks_.emplace_back(tmp_);
        }
        Eigen::MatrixXd grad_softmax_ = Eigen::MatrixXd::Zero(output_.rows() * output_.cols(),
                                                              output_.rows() * output_.cols());
        for (uint h_ = 0; h_ < output_.cols(); ++h_) {
            for (uint v_ = 0; v_ < output_.cols(); ++v_) {
                std::vector<double> to_diag_;
                to_diag_.reserve(output_.rows());
                for_each(blocks_.cbegin(),
                         blocks_.cend(),
                         [&to_diag_, h_, v_](const Eigen::MatrixXd &m)->void {
                             to_diag_.emplace_back(m(h_,
                                                     v_));
                         });
                Eigen::MatrixXd block = Eigen::VectorXd::Map(to_diag_.data(),
                                                             to_diag_.size()).asDiagonal();
                /*分块梯度拼接*/
                grad_softmax_.block(h_ * output_.cols(),
                                    v_ * output_.cols(),
                                    output_.cols(),
                                    output_.cols()) = block;
            }
        }
        grad_softmax_ = grad_softmax_.transpose().eval();/*需要转置*/
        /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
        _gradientOfOutput.emplace(output_._name,
                                  grad_w_ * grad_softmax_);
        /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
        auto has_ = input._gradientOfOperator.find(_name);
        if (has_ == input._gradientOfOperator.end()) {
            input._gradientOfOperator.emplace(_name,
                                              grad_input_ * grad_softmax_);
        }
        else {
            has_->second += grad_input_ * grad_softmax_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
        }
    }
    else if (_act_func == "sigmoid") {
        /*z=sigmoid(y)*/
        output_._value = sigmoid(output_._value);
        /*z对y的梯度*/
        Eigen::MatrixXd to_diag_ = output_._value.array() - output_._value.array().pow(2);
        Eigen::MatrixXd grad_sigmoid_ = Eigen::VectorXd::Map(to_diag_.data(),
                                                             to_diag_.size()).asDiagonal();
        /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
        _gradientOfOutput.emplace(output_._name,
                                  grad_w_ * grad_sigmoid_);
        /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
        auto has_ = input._gradientOfOperator.find(_name);
        if (has_ == input._gradientOfOperator.end()) {
            input._gradientOfOperator.emplace(_name,
                                              grad_input_ * grad_sigmoid_);
        }
        else {
            has_->second += grad_input_ * grad_sigmoid_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
        }
    }
    else if (_act_func == "tanh") {
        /*z=tanh(y)*/
        output_._value = 2 * sigmoid(2 * output_._value).array() - 1;
        /*z对y的梯度*/
        Eigen::MatrixXd to_diag_ = 1 - output_._value.array() * output_._value.array();
        Eigen::MatrixXd grad_tanh_ = Eigen::VectorXd::Map(to_diag_.data(),
                                                          to_diag_.size()).asDiagonal();
        /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
        _gradientOfOutput.emplace(output_._name,
                                  grad_w_ * grad_tanh_);
        /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
        auto has_ = input._gradientOfOperator.find(_name);
        if (has_ == input._gradientOfOperator.end()) {
            input._gradientOfOperator.emplace(_name,
                                              grad_input_ * grad_tanh_);
        }
        else {
            has_->second += grad_input_ * grad_tanh_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
        }
    }
    else {
        /*z=y=x*w+b*/
        /*z对w的梯度 = y对w的梯度*/
        _gradientOfOutput.emplace(output_._name,
                                  grad_w_);
        /*z对x的梯度 = y对x的梯度*/
        auto has_ = input._gradientOfOperator.find(_name);
        if (has_ == input._gradientOfOperator.end()) {
            input._gradientOfOperator.emplace(_name,
                                              grad_input_);
        }
        else {
            has_->second += grad_input_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
        }
    }
}

void FullConnect::backward() {
    for (const auto &ele :_inputOutputPair) {
        const auto &output_name_{ele.second};
        /*loss对z的梯度*/
        const auto &grad_loss_to_output_{_graphManager._variables[output_name_]->_gradientOfLoss};
        /*loss对w的梯度 = z对w的梯度 * loss对z的梯度*/
        Eigen::MatrixXd tmp_ = _gradientOfOutput[output_name_] * grad_loss_to_output_;
        if (_gradWeightAndBias.size() == 0) {
            _gradWeightAndBias = tmp_;
        }
        else {
            _gradWeightAndBias += tmp_;
        }
        /*loss对x的梯度 = z对x的梯度 * loss对z的梯度*/
        const auto &input_name_{ele.first};
        tmp_ = _graphManager._variables[input_name_]->_gradientOfOperator[_name] * grad_loss_to_output_;
        auto &grad_input_ = _graphManager._variables[input_name_]->_gradientOfLoss;
        if (grad_input_.size() == 0) {
            grad_input_ = tmp_;
        }
        else {
            grad_input_ += tmp_;
        }
    }

}