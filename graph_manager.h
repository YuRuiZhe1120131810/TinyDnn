/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#ifndef TINYDNN__GRAPH_MANAGER_H
#define TINYDNN__GRAPH_MANAGER_H
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "variable.h"
#include "operator_base.h"

class OperatorBase;
class Variable;
class GraphManager {
    /*图管理器记录Variable Operator 记录Variable是被哪个Operator输出的 被哪个Operator调用的
     * Variable被Operator调用 Variable.name为key value是list 每个元素是Operator.name
     * Variable被Operator输出 Variable.name为key value是Operator.name
     * Variable与Operator被记录在map里 方便查找
     * 使用单例模式 全局唯一*/
private:

public:
    /*存储栈上自动变量的指针 要注意自动变量的生命周期 防止对悬空指针解引用*/
    std::unordered_map<std::string, OperatorBase *> _operators;
    std::unordered_map<std::string, std::vector<std::string>> _variableCallBy;
    std::unordered_map<std::string, std::string> _variableCreateBy;
    /*存储栈上自动变量的指针 要注意自动变量的生命周期 防止对悬空指针解引用*/
    std::unordered_map<std::string, Variable *> _variables;
    ~GraphManager() {
        _operators.clear();
        _variableCallBy.clear();
        _variableCreateBy.clear();
        _variables.clear();
    };
    GraphManager(const GraphManager &) = delete;
    GraphManager &operator=(const GraphManager &) = delete;
    static GraphManager &get() {
        thread_local static GraphManager instance_;
        return instance_;
    }
    GraphManager() {
        _operators.clear();
        _variableCallBy.clear();
        _variableCreateBy.clear();
        _variables.clear();
    }
    void backward(Variable &) const;
    void update() const;
};

#endif //TINYDNN__GRAPH_MANAGER_H
