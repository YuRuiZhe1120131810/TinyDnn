/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 每个Variable扇出为1 只能作为一个LAYER的输入 不能作为多个LAYER的输入
*/

#include "dnn.h"

#include <utility>
namespace yrz {

    Variable &Variable::operator=(Variable &&other) noexcept {
        if (this != &other) {
            _row = other._row, _col = other._col, _val = other._val, _gen_by = other._gen_by;
            other._gen_by = nullptr;
            other.clear();
        }
        return *this;
    }

    Variable::Variable(const uint64_t &row,
                       const uint64_t &col,
                       std::shared_ptr<LAYER> l) : _row(row),
                                                   _col(col),
                                                   _gen_by(std::move(l)) {
        std::vector<double> tmp_(_row * _col,
                                 0);
        _val = Eigen::MatrixXd::Map(&tmp_[0],
                                    _row,
                                    _col);
        _grad = Eigen::MatrixXd::Map(&tmp_[0],
                                     _row,
                                     _col);
    }

    void Variable::set_val(const Eigen::MatrixXd &new_val) {
        assert(new_val.rows() == _row && new_val.cols() == _col);
        _val = new_val;
    }

    void Variable::clear() {
        _row = 0, _col = 0;
        _val.resize(0,
                    0);
        _grad.resize(0,
                     0);
    }

    FullConnect::FullConnect(uint64_t in_channel,
                             uint64_t out_channel,
                             double learning_rate,
                             const std::string act_func) : _learning_rate{learning_rate},
                                                           _act_func{std::move(act_func)} {
        _weightAndBias = Variable(in_channel + 1,
                                  out_channel,
                                  std::make_shared<LAYER>(*this));
    }

    void FullConnect::initWeight(uint64_t seed) {
        srand(seed);
        _weightAndBias.set_val(Eigen::MatrixXd::Random(_weightAndBias.get_val().rows(),
                                                       _weightAndBias.get_val().cols()));
    }

    std::shared_ptr<Variable> FullConnect::forward(std::shared_ptr<Variable> &input_ptr) {
        assert(input_ptr->_col + 1 == _weightAndBias._row);
        _input_ptr = input_ptr;
        /*扩大一个维度*/
        Eigen::MatrixXd bias_coef_ = Eigen::MatrixXd::Ones(_input_ptr->_row,
                                                           1);
        Eigen::MatrixXd input_expand_(_input_ptr->_row,
                                      _input_ptr->_col + 1);
        input_expand_ << _input_ptr->get_val(), bias_coef_;
        /*计算y=xw+b*/
        auto output_ptr_ = std::make_shared<Variable>(Variable(input_expand_ * _weightAndBias.get_val(),
                                                               std::make_shared<LAYER>(*this)));
        /*填充节点连接关系*/
        output_ptr_->add_predecessor(_input_ptr);
        _input_ptr->add_successor(output_ptr_);
        /*y对w的梯度*/
        Eigen::MatrixXd identity_ = Eigen::MatrixXd::Identity(_weightAndBias._col,
                                                              _weightAndBias._col);
        Eigen::MatrixXd grad_w_ = kroneckerProduct(identity_,
                                                   input_expand_).transpose().eval();
        /*y对x的梯度*/
        identity_ = Eigen::MatrixXd::Identity(_input_ptr->_row,
                                              _input_ptr->_row);
        Eigen::MatrixXd grad_input_ = kroneckerProduct(_weightAndBias.get_val().transpose(),
                                                       identity_).transpose().eval();
        auto sigmoid = [&](const Eigen::MatrixXd &x)->Eigen::MatrixXd { return 1 / ((-x).array().exp() + 1); };
        /*激活函数*/
        if (_act_func == "relu") {
            /*z=relu(y)*/
            output_ptr_->set_val(output_ptr_->get_val().cwiseMax(0));
            /*z对y的梯度*/
            Eigen::MatrixXd to_diag_ = Eigen::VectorXd::Map(output_ptr_->get_val().data(),
                                                            output_ptr_->get_val().size()).asDiagonal();
            Eigen::MatrixXd grad_relu_ = to_diag_.array().sign();
            /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
            _weightAndBias.set_grad(grad_w_ * grad_relu_);
            /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
            _input_ptr->set_grad(grad_input_ * grad_relu_);
        }
        else if (_act_func == "softmax") {
            assert(_weightAndBias._col == 1);/*softmax不是逐元素函数 要求输入输出都是向量*/
            /*z=softmax(y)*/
            Eigen::MatrixXd max_value_ = Eigen::MatrixXd::Ones(output_ptr_->get_val().rows(),
                                                               output_ptr_->get_val().cols())
                * output_ptr_->get_val().maxCoeff();
            Eigen::MatrixXd output_val_ = (output_ptr_->get_val() - max_value_).array().exp().matrix();/*防止浮点溢出*/
            const double denominator = output_val_.array().sum();
            output_ptr_->set_val(output_val_ / denominator);
            /*z对y的梯度*/
            Eigen::MatrixXd grad_softmax_ = -1 * output_ptr_->get_val() * output_ptr_->get_val().transpose();
            assert(grad_softmax_.rows() == grad_softmax_.cols() && grad_softmax_.rows() == output_ptr_->_row);
            grad_softmax_.diagonal().setZero();
            Eigen::MatrixXd tmp_ = Eigen::VectorXd::Map(output_ptr_->get_val().data(),
                                                        output_ptr_->get_val().size()).asDiagonal();
            grad_softmax_ += tmp_ * (Eigen::MatrixXd::Ones(tmp_.rows(),
                                                           tmp_.cols()) - tmp_);
            /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
            _weightAndBias.set_grad(grad_w_ * grad_softmax_);
            /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
            _input_ptr->set_grad(grad_input_ * grad_softmax_);
        }
        else if (_act_func == "sigmoid") {
            /*z=sigmoid(y)*/
            output_ptr_->set_val(sigmoid(output_ptr_->get_val()));
            /*z对y的梯度*/
            Eigen::MatrixXd to_diag_ = output_ptr_->get_val().array() *
                (Eigen::MatrixXd::Ones(output_ptr_->_row,
                                       output_ptr_->_col) - output_ptr_->get_val()).array();
            Eigen::MatrixXd grad_sigmoid_ = Eigen::VectorXd::Map(to_diag_.data(),
                                                                 to_diag_.size()).asDiagonal();
            /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
            _weightAndBias.set_grad(grad_w_ * grad_sigmoid_);
            /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
            _input_ptr->set_grad(grad_input_ * grad_sigmoid_);
        }
        else if (_act_func == "tanh") {
            /*z=tanh(y)*/
            output_ptr_->set_val(2 * sigmoid(2 * output_ptr_->get_val())
                                     - Eigen::MatrixXd::Ones(output_ptr_->_row,
                                                             output_ptr_->_col));
            /*z对y的梯度*/
            Eigen::MatrixXd to_diag_ = 1 - output_ptr_->get_val().array() * output_ptr_->get_val().array();
            Eigen::MatrixXd grad_tanh_ = Eigen::VectorXd::Map(to_diag_.data(),
                                                              to_diag_.size()).asDiagonal();
            /*z对w的梯度 = y对w的梯度 * z对y的梯度*/
            _weightAndBias.set_grad(grad_w_ * grad_tanh_);
            /*z对x的梯度 = y对x的梯度 * z对y的梯度*/
            _input_ptr->set_grad(grad_input_ * grad_tanh_);
        }
        else {
            /*z=y=x*w+b*/
            /*z对w的梯度 = y对w的梯度*/
            _weightAndBias.set_grad(grad_w_);
            /*z对x的梯度 = y对x的梯度*/
            _input_ptr->set_grad(grad_input_);
        }
        return output_ptr_;
    }

    void FullConnect::backward(std::shared_ptr<Variable> &input_ptr) {
        if(_input_ptr)
        /*最终loss对输入的梯度*/
        Eigen::MatrixXd grad_wrt_loss_ = (_input_ptr->get_grad() * _output.get_grad()).transpose();
        assert(grad_wrt_loss_.rows() == _input->_row && grad_wrt_loss_.cols() == _input->_col);
        _input->set_grad(grad_wrt_loss_);
        /*更新参数*/
        _weightAndBias.set_val(_weightAndBias.get_grad() * _learning_rate * -1.0 + _weightAndBias.get_val());
        /*梯度重置*/
        _weightAndBias.set_grad(_weightAndBias.get_grad() * 0.0);
    }

    void FullConnect::update() {

    }

    Variable &ReduceSum::forward(yrz::Variable &input) {
        _input = &input;
        _output = Variable(1,
                           1,
                           this);
        std::vector<double> tmp_{input.get_val().sum()};
        Eigen::MatrixXd output_val_ = Eigen::MatrixXd::Map(&tmp_[0],
                                                           1,
                                                           1);
        _output.set_val(output_val_);
        /*输出对输入的梯度*/
        input.set_grad(Eigen::MatrixXd::Ones(input._row,
                                             input._col));
        return _output;
    }
    void ReduceSum::backward() {
        /*最终loss对输入的梯度*/
        assert(_input->get_grad().cols() == _output.get_grad().rows());
        Eigen::MatrixXd grad_wrt_loss_ = _input->get_grad() * _output.get_grad();
        assert(grad_wrt_loss_.rows() == _input->_row && grad_wrt_loss_.cols() == _input->_col);
        _input->set_grad(grad_wrt_loss_);
        /*调用上游节点*/
        _input->_gen_by->backward();
    }

    Variable &MSELoss::forward(Variable &input,
                               Variable &label) {
        assert(input._row == label._row && input._col == label._col);
        _input = &input;
        _output = Variable(input._row,
                           input._col,
                           this);
        Eigen::MatrixXd tmp_ = input.get_val() - label.get_val();
        double loss_val_ = (tmp_.transpose() * tmp_).trace();
        _output.set_val(Eigen::MatrixXd::Identity(1,
                                                  1) * loss_val_);
        /*输出对输入与label的梯度*/
        input.set_grad(2 * (input.get_val() - label.get_val()));
        label.set_grad(2 * (label.get_val() - input.get_val()));
        return _output;
    }
    void MSELoss::backward() {
        _output.set_grad(Eigen::MatrixXd::Identity(1,
                                                   1));
        _input->_gen_by->backward();
    }

    Variable &CrossEntropyLoss::forward(Variable &input,
                                        Variable &label) {
        assert(input.get_val().maxCoeff() > 0
                   && label.get_val().maxCoeff() <= 1.0
                   && label.get_val().minCoeff() >= 0.0);
        std::cout << "CrossEntropyLoss input=\n" << input.get_val() << std::endl
                  << "label=\n" << label.get_val() << std::endl;

        _input = &input;
        _output = Variable(1,
                           1,
                           this);
        double output_val_ = (-1.0 * label.get_val().array() * input.get_val().array().log()).sum();
        _output.set_val(Eigen::MatrixXd::Identity(1,
                                                  1) * output_val_);
        /*输出对输入的梯度*/
        std::cout << "CrossEntropyLoss grad loss wrt input=\n"
                  << -1.0 * label.get_val().array() * input.get_val().array().inverse() << std::endl;
        input.set_grad(-1.0 * label.get_val().array() * input.get_val().array().inverse());
        return _output;
    }
    void CrossEntropyLoss::backward() {
        _output.set_grad(Eigen::MatrixXd::Identity(1,
                                                   1));
        _input->_gen_by->backward();
    }
}
