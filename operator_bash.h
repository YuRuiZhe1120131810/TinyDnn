/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 每个Variable扇出为1 只能作为一个LAYER的输入 不能作为多个LAYER的输入
*/
#ifndef EIGEN__OPERATORBASE_H
#define EIGEN__OPERATORBASE_H
#include <string>
class OperatorBase {
    /*Operator是计算图的边 前馈计算输入输出是Variable 可以多次前馈计算 每次前馈计算会输出一个新Variable 记录自身的参数 记录输入输出pair
     * 反馈计算读取输出Variable的梯度 并计算赋值更新输入Variable的梯度*/
    std::string _name;
    static uint32_t _forward_cnt;
};
#endif //EIGEN__OPERATORBASE_H
