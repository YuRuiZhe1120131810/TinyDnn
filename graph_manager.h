/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
 * 每个Variable扇出为1 只能作为一个LAYER的输入 不能作为多个LAYER的输入
*/
#ifndef EIGEN__GRAPHMANAGER_H
#define EIGEN__GRAPHMANAGER_H
#include <string>
#include <unordered_map>
#import "Variable.h"
class GraphManager {
    /*图管理器记录Variable Operator 记录Variable是被哪个Operator输出的 被哪个Operator调用的*/

};
#endif //EIGEN__GRAPHMANAGER_H
