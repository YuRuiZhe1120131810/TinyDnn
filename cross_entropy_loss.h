/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 普通常量名 大写字母开头 下划线结尾
*/
#ifndef TINYDNN__CROSS_ENTROPY_LOSS_H
#define TINYDNN__CROSS_ENTROPY_LOSS_H
#include "operator_base.h"
class OperatorBase;
class Variable;
class GraphManager;
class CrossEntropyLoss : public OperatorBase {
    static uint _instanceCount;/*记录有多少个CrossEntropyLoss对象*/
    uint _forwardCount;/*记录forward次数*/
    explicit CrossEntropyLoss(GraphManager &graph_manager);
    Variable forward(Variable &prediction,
                     Variable &label) override;
    void backward() override;
};
uint CrossEntropyLoss::_instanceCount{0};
#endif //TINYDNN__CROSS_ENTROPY_LOSS_H
