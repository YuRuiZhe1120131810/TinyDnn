/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/

#ifndef EIGEN__DNN_H
#define EIGEN__DNN_H
#include <bootstrap.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Geometry>
#include <utility>
#include <vector>
#include <list>
#include <unordered_set>
#include <iostream>
#include <random>
#include <memory>
#include <queue>

namespace yrz {
    class FullConnect;
    class ReduceSum;
    class Softmax;
    class MSELoss;
    class CrossEntropyLoss;
    enum LAYER { FullConnect, ReduceSum, MSELoss, CrossEntropyLoss };

    struct Variable {
        uint64_t _row = 0, _col = 0;/*矩阵的行与列*/
        Eigen::MatrixXd _val, _grad;/*最终loss对本Variable的梯度*/
        std::shared_ptr<LAYER> _gen_by;/*产生本Variable的layer*/
        std::list<std::shared_ptr<LAYER>> _used_by;/*调用本Variable的layer*/
        std::list<std::shared_ptr<Variable>> _predecessors;/*前驱节点*/
        std::list<std::shared_ptr<Variable>> _successors;/*后继节点*/
        explicit Variable(const uint64_t &row = 0,
                          const uint64_t &col = 0,
                          std::shared_ptr<LAYER> l = nullptr);
        explicit Variable(const Eigen::MatrixXd &new_val,
                          std::shared_ptr<LAYER> l = nullptr) : _row(new_val.rows()),
                                                                _col(new_val.cols()),
                                                                _val(new_val),
                                                                _gen_by(std::move(l)) {}
        /*禁用拷贝构造，根据Variable构造另一个Variable是无意义的，构造计算图节点不应当依赖另一个节点*/
        Variable(const Variable &other) = delete;
        /*禁用拷贝赋值，拿Variable给已初始化的Variable赋值是无意义的，计算图节点不能覆盖另一个节点*/
        Variable &operator=(const Variable &other) = delete;
        /*移动赋值*/
        Variable &operator=(Variable &&other) noexcept;
        /*析构*/
        ~Variable() = default;
        const Eigen::MatrixXd &get_val() const { return _val; }
        const Eigen::MatrixXd &get_grad() const { return _grad; }
        void set_val(const Eigen::MatrixXd &new_val);
        void set_grad(const Eigen::MatrixXd &new_grad) { _grad = new_grad; }
        void clear();/*清空_val与_grad*/
        void add_predecessor(const std::shared_ptr<Variable> &pred) { _predecessors.emplace_back(pred); }
        void add_successor(const std::shared_ptr<Variable> &succ) { _successors.emplace_back(succ); }
    };

    class FullConnect {
        /*全连接层*/
    public:
        explicit FullConnect(uint64_t in_channel = 0,
                             uint64_t out_channel = 0,
                             double learning_rate = 0.0001,
                             const std::string act_func = "relu");/*激活函数relu softmax sigmoid tanh*/
        Variable _weightAndBias;
        std::shared_ptr<Variable> _input_ptr;/*forward时记录input信息*/
        const double _learning_rate;/*学习率*/
        const std::string _act_func;/*激活函数*/
        std::shared_ptr<Variable> forward(std::shared_ptr<Variable> &input_ptr);
        void backward(std::shared_ptr<Variable> &input_ptr);
        void update();/*更新参数*/
        bool loadWeight() { return true; };
        bool saveWeight() { return true; };
        void initWeight(uint64_t seed = 0);
    };

    class ReduceSum {
        /*全加和*/
    public:
        Variable &forward(Variable &input);
        void backward(Variable &input);/*backward的input是forward的output*/
        std::shared_ptr<Variable> _input;
    };

    class Relu {
    public:
        Variable &forward(Variable &input);
        void backward();
        Variable *_input, _output;
    };

    class Softmax {
    public:
        Variable &forward(Variable &input);
        void backward();
        Variable *_input, _output;
    };

    class MSELoss {
    public:
        Variable &forward(Variable &input,
                          Variable &label);
        void backward();
        Variable *_input, _output;
    };

    class CrossEntropyLoss {
    public:
        Variable &forward(Variable &input,
                          Variable &label);
        void backward();
        Variable *_input, _output;
    };

    class Concatenate {
        /*拼接层*/
        Variable &forward(Variable &input,
                          Variable &label);
        void backward();
        Variable _output;
        std::vector<std::shared_ptr<Variable>> _input;
    };
}
#endif //EIGEN__DNN_H
