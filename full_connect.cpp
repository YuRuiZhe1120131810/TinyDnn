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

uint32_t FullConnect::_instanceCount{0};

FullConnect::FullConnect(uint64_t in_channel,
                         uint64_t out_channel,
                         double learning_rate,
                         std::string act_func,
                         GraphManager &graph_manager) : _act_func(std::move(act_func)),
                                                        _learning_rate(learning_rate),
                                                        _forwardCount(0),
                                                        _weight(Eigen::MatrixXd::Ones(in_channel,
                                                                                      out_channel) * 0.1),
                                                        _bias(Eigen::MatrixXd::Ones(1,
                                                                                    out_channel) * 0.1),
                                                        OperatorBase(graph_manager) {
    _name = std::string("FullConnect_").append(std::to_string(_instanceCount++));/*构造实例计数增一*/
    _graphManager._operators.emplace(_name,
                                     std::shared_ptr<FullConnect>(this));
}

Variable FullConnect::forward(Variable &input) {
    assert(input.cols() == _weight.rows() && "输入维度需要匹配");
    /*计算y=xw+b*/
    auto output_ = Variable(input.val() * _weight + _bias.replicate(input.rows(),
                                                                    1),
                            _name + "_" + std::to_string(_forwardCount++),
                            _graphManager);
    std::cout << "layer=" << _name
              << ",x," << input.val().rows() << "行" << input.val().cols() << "列"
              << ",w," << _weight.rows() << "行" << _weight.cols() << "列"
              << ",b," << _bias.rows() << "行" << _bias.cols() << "列"
              << ",y," << output_.val().rows() << "行" << output_.val().cols() << "列" << std::endl;
    /*填充节点连接关系*/
    _graphManager._variableCreateBy.emplace(output_._name,
                                            _name);
    _graphManager._variableCallBy[input._name].emplace_back(_name);
    _inputOutputPair.emplace_back(std::make_pair(input._name,
                                                 output_._name));
    /*y对w的梯度*/
    Eigen::MatrixXd identity_ = Eigen::MatrixXd::Identity(_weight.cols(),
                                                          _weight.cols());
    Eigen::MatrixXd grad_w_ = kroneckerProduct(identity_,
                                               input.val().transpose()).eval();
    std::cout << "layer=" << _name << ",y对w的梯度" << grad_w_.rows() << "行" << grad_w_.cols() << "列" << std::endl;
    /*y对x的梯度*/
    identity_ = Eigen::MatrixXd::Identity(input.rows(),
                                          input.rows());
    Eigen::MatrixXd grad_input_ = kroneckerProduct(_weight,
                                                   identity_).eval();
    std::cout << "layer=" << _name << ",y对x的梯度" << grad_input_.rows() << "行" << grad_input_.cols() << "列" << std::endl;
    /*y对bias的梯度恒为1 但是随着input的行数变化 bias的梯度的行数也在变化 可能需要把梯度累加到一行 导致梯度更新不准*/
    auto sigmoid = [&](const Eigen::MatrixXd &x)->Eigen::MatrixXd { return 1 / ((-x).array().exp() + 1); };
    /*激活函数*/
    if (_act_func == "relu") {
        /*z=relu(y)*/
        output_._value = output_._value.cwiseMax(0);
        /*z对y的梯度*/
        Eigen::MatrixXd to_diag_ = Eigen::VectorXd::Map(output_._value.data(),
                                                        output_._value.size()).asDiagonal();
        Eigen::MatrixXd grad_relu_ = to_diag_.array().sign();
        std::cout << "layer=" << _name << ",relu对y的梯度" << grad_relu_.rows() << "行" << grad_relu_.cols() << "列"
                  << std::endl;
        /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
        _wGradOfOutput.emplace(output_._name,
                               grad_w_ * grad_relu_);
        std::cout << "layer=" << _name << ",relu对w的梯度" << _wGradOfOutput[output_._name].rows() << "行"
                  << _wGradOfOutput[output_._name].cols() << "列" << std::endl;
        /*z对bias的梯度 = y对bias的梯度 * z对y的梯度*/
        _bGradOfOutput.emplace(output_._name,
                               grad_relu_);
        std::cout << "layer=" << _name << ",relu对bias的梯度" << _bGradOfOutput[output_._name].rows() << "行"
                  << _bGradOfOutput[output_._name].cols() << "列" << std::endl;
        /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
        auto has_ = input._gradientOfOperator.find(_name);
        if (has_ == input._gradientOfOperator.end()) {
            input._gradientOfOperator.emplace(_name,
                                              grad_input_ * grad_relu_);
        }
        else {
            has_->second += grad_input_ * grad_relu_;/*如果一个Variable被同一layer多次前馈 梯度应该累加*/
        }
        std::cout << "layer=" << _name << ",relu对x的梯度" << input._gradientOfOperator[_name].rows() << "行"
                  << input._gradientOfOperator[_name].cols() << "列" << std::endl;
    }
    else if (_act_func == "softmax") {
        /*softmax不是逐元素函数 是逐行函数 每一行之间的计算结果是完全独立的*/
        /*z=softmax(y)*/
        Eigen::MatrixXd max_value_ = output_._value.rowwise().maxCoeff().replicate(1,
                                                                                   output_.cols());
        Eigen::MatrixXd output_val_ = (output_._value - max_value_).array().exp().matrix();/*防止浮点溢出*/
        /*逐行均一化*/
        Eigen::MatrixXd denominator_ = output_val_.rowwise().lpNorm<1>().array().pow(-1).replicate(1,
                                                                                                   output_val_.cols());
        output_._value = output_val_.cwiseProduct(denominator_);
        /*z对y的梯度 需要由多个分块对角矩阵拼接而成 z的每一行都计算一个梯度矩阵 按位置取梯度矩阵元素拼接成对角矩阵*/
        std::vector<Eigen::MatrixXd> blocks_;
        blocks_.reserve(output_.rows());
        for (const auto row : output_._value.rowwise()) {
            Eigen::MatrixXd tmp_ = row.transpose() * row * -1;
            tmp_.diagonal().setZero();
            const Eigen::MatrixXd diag_ = row.asDiagonal();
            tmp_ += diag_ - diag_ * diag_;
            blocks_.emplace_back(tmp_);
        }
        Eigen::MatrixXd grad_softmax_ = Eigen::MatrixXd::Zero(blocks_.size() * output_.cols(),
                                                              blocks_.size() * output_.cols());
        for (uint32_t h_ = 0; h_ < output_.cols(); ++h_) {
            for (uint32_t v_ = 0; v_ < output_.cols(); ++v_) {
                std::vector<double> to_diag_;
                to_diag_.reserve(output_.rows());
                for_each(blocks_.cbegin(),
                         blocks_.cend(),
                         [&to_diag_, h_, v_](const Eigen::MatrixXd &m)->void {
                             to_diag_.emplace_back(m(h_,
                                                     v_));
                         });
                Eigen::MatrixXd blk_ = Eigen::VectorXd::Map(to_diag_.data(),
                                                            to_diag_.size()).asDiagonal();
                /*分块梯度拼接*/
                grad_softmax_.block(h_ * blk_.rows(),
                                    v_ * blk_.cols(),
                                    blk_.cols(),
                                    blk_.cols()) = blk_;
            }
        }
        grad_softmax_ = grad_softmax_.transpose().eval();/*需要转置*/
        /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
        _wGradOfOutput.emplace(output_._name,
                               grad_w_ * grad_softmax_);
        /*z对bias的梯度 = y对bias的梯度 * z对y的梯度*/
        _bGradOfOutput.emplace(output_._name,
                               grad_softmax_);
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
        _wGradOfOutput.emplace(output_._name,
                               grad_w_ * grad_sigmoid_);
        /*z对bias的梯度 = y对bias的梯度 * z对y的梯度*/
        _bGradOfOutput.emplace(output_._name,
                               grad_sigmoid_);
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
        _wGradOfOutput.emplace(output_._name,
                               grad_w_ * grad_tanh_);
        /*z对bias的梯度 = y对bias的梯度 * z对y的梯度*/
        _bGradOfOutput.emplace(output_._name,
                               grad_tanh_);
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
        _wGradOfOutput.emplace(output_._name,
                               grad_w_);
        /*z对bias的梯度 = y对bias的梯度*/
        _bGradOfOutput.emplace(output_._name,
                               Eigen::MatrixXd::Identity(output_.rows() * output_.cols(),
                                                         output_.rows() * output_.cols()));
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
    return output_;
}

void FullConnect::backward() {
    for (const auto &ele :_inputOutputPair) {
        const auto &output_name_{ele.second};
        /*loss对z的梯度*/
        const auto &grad_loss_to_output_{_graphManager._variables.at(output_name_)->_gradientOfLoss};
        /*loss对w的梯度 = z对w的梯度 * loss对z的梯度*/
        Eigen::MatrixXd tmp_ = _wGradOfOutput.at(output_name_) * grad_loss_to_output_;
        tmp_ = tmp_.reshaped(_weight.rows(),
                             _weight.cols());
        if (_gradWeight.size() == 0) {
            _gradWeight = tmp_;
        }
        else {
            _gradWeight += tmp_;
        }
        /*loss对b的梯度 = z对bias的梯度 * loss对z的梯度*/
        tmp_ = _bGradOfOutput.at(output_name_) * grad_loss_to_output_;
        /*bias是一个只有1行的向量 前馈时复制自身 反馈时调整形状使bias_grad的column=bias的column
         * 最后累加成一个1行向量*/
        tmp_ = tmp_.reshaped(tmp_.rows() / _bias.cols(),
                             _bias.cols()).colwise().sum();
        if (_gradBias.size() == 0) {
            _gradBias = tmp_;
        }
        else {
            _gradBias += tmp_;
        }
        /*loss对x的梯度 = z对x的梯度 * loss对z的梯度*/
        const auto &input_name_{ele.first};
        tmp_ = _graphManager._variables.at(input_name_)->_gradientOfOperator.at(_name) * grad_loss_to_output_;
        tmp_ = tmp_.reshaped(_graphManager._variables[input_name_]->rows(),
                             _graphManager._variables[input_name_]->cols());
        auto &grad_loss_to_input_ = _graphManager._variables[input_name_]->_gradientOfLoss;
        if (grad_loss_to_input_.size() == 0) {
            grad_loss_to_input_ = tmp_;
        }
        else {
            grad_loss_to_input_ += tmp_;
        }
        std::cout << "aaaaa" << std::endl << _graphManager._variables[input_name_]->_gradientOfLoss << std::endl;
    }
}

void FullConnect::update() {
    std::cout << "layer=" << _name << ",_weight=" << std::endl << _weight << std::endl
              << "_gradWeight=" << std::endl << _gradWeight << std::endl;
    std::cout << "layer=" << _name << ",_bias=" << std::endl << _bias << std::endl
              << "_gradBias=" << std::endl << _gradBias << std::endl;
    _weight += _gradWeight * _learning_rate;
    _bias += _gradBias * _learning_rate;
}