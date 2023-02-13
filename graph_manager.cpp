/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include "graph_manager.h"
#include "variable.h"
#include "operator_base.h"
#include <queue>
#include <list>
#include <iostream>

GraphManager GraphManager::_instance;

void GraphManager::backward(Variable &start) {
    /*从loss节点开始BFS遍历 要求graph是树形结构 把loss节点的梯度回传给所有Operator和Variable*/
    std::queue<std::string> fifo_;
    fifo_.emplace(start._name);
    while (!fifo_.empty()) {
        std::cout << "当前Variable=" << fifo_.front();
        const auto &iter = _variableCreateBy.find(fifo_.front());
        fifo_.pop();
        if (iter == _variableCreateBy.end()) {
            std::cout << "是叶子节点,continue" << std::endl;
            continue;
        }
        std::cout << "被Operator=" << iter->second << "调用" << std::endl;
        std::shared_ptr<OperatorBase> &invoke_op_ = _operators.at(iter->second);
        invoke_op_->backward();
        for (const auto &io_var_name_ :invoke_op_->_inputOutputPair) {
            fifo_.emplace(io_var_name_.first);
            std::cout << "Operator=" << invoke_op_->_name
                      << "的输入Variable=" << io_var_name_.first << std::endl;
        }
    }
}

void GraphManager::update() {
    for (const auto &pair: _operators) {
        pair.second->update();/*每个op都是初始化状态，相应的梯度和参数没有记录到_operators中*/
    }
}