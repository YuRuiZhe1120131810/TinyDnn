/*
 * 类名 大写字母开头 驼峰法
 * 方法名 小写字母开头 驼峰法
 * 常量成员名 下划线开头 第一个字母大写 驼峰法
 * 变量成员名 下划线开头 第一个字母小写 驼峰法
 * 参数名 小写字母开头 所有字母小写 下划线分隔法
 * 普通变量名 小写字母开头 下划线结尾
*/
#include "utility.h"

Eigen::MatrixXd Utility::oneHot(const Eigen::MatrixXd &m,
                                const bool row_wise) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m.rows(),
                                                   m.cols());
    Eigen::MatrixXf::Index idx_;
    for (int r = 0; row_wise && r < m.rows(); ++r) {
        m.row(r).maxCoeff(&idx_);
        result.coeffRef(r,
                        idx_) = 1;
    }
    for (int c = 0; !row_wise && c < m.cols(); ++c) {
        m.col(c).maxCoeff(&idx_);
        result.coeffRef(idx_,
                        c) = 1;
    }
    return result;
}

Eigen::MatrixXd Utility::matSame(const Eigen::MatrixXd &a,
                                 const Eigen::MatrixXd &b,
                                 const bool row_wise) {
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(a.rows(),
                                                   b.cols());
    for (int r = 0; row_wise && r < a.rows(); ++r) {
        result.coeffRef(r,
                        0) = a.row(r).isApprox(b.row(r));
    }
    for (int c = 0; !row_wise && c < b.cols(); ++c) {
        result.coeffRef(0,
                        c) = a.col(c).isApprox(b.col(c));
    }
    return row_wise ? Eigen::MatrixXd(result.rowwise().sum()) : Eigen::MatrixXd(result.colwise().sum());
}