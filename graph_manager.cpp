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

void GraphManager::backward(Variable &start) const {
    /*从loss节点开始BFS遍历 要求graph是树形结构 把loss节点的梯度回传给所有Operator和Variable*/
    std::queue<std::string> fifo_;
    fifo_.emplace(start._name);
    while (!fifo_.empty()) {
        const auto &iter = _variableCreateBy.find(fifo_.front());
        fifo_.pop();
        if (iter == _variableCreateBy.end()) {
            continue;
        }
        OperatorBase *invoke_op_ = _operators.at(iter->second);
        invoke_op_->backward();
        for (const auto &io_var_name_ :invoke_op_->_inputOutputPair) {
            fifo_.emplace(io_var_name_.first);
        }
    }
}

void GraphManager::update() const {
    for (const auto &pair: _operators) {
        pair.second->update();
    }
}

void GraphManager::release() {
    _variableCallBy.clear();
    _variables.clear();
    _variableCreateBy.clear();
    for (auto &pair:_operators) {
        pair.second->release();
    }
}