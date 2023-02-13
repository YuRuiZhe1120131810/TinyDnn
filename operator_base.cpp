/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include "operator_base.h"
#include "variable.h"
class Variable;
OperatorBase::OperatorBase(GraphManager &gm) : _graphManager{gm} {
    _inputOutputPair.clear();
}
OperatorBase::~OperatorBase() {
    _inputOutputPair.clear();
}
Variable OperatorBase::forward(Variable &v) {
    assert(false && "operator_base not inplement");
}
Variable OperatorBase::forward(Variable &v1,
                               Variable &v2) {
    assert(false && "operator_base not inplement");
}
void OperatorBase::backward() {
    assert(false && "operator_base not inplement");
}
void OperatorBase::update() {
    assert(false && "operator_base not inplement");
}
bool OperatorBase::loadWeight() {
    assert(false && "operator_base not inplement");
}
bool OperatorBase::saveWeight() {
    assert(false && "operator_base not inplement");
}
void OperatorBase::initWeight(uint64_t seed) {
    assert(false && "operator_base not inplement");
}

