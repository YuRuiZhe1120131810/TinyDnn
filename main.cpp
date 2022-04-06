#include <iostream>
#include "dnn.h"

int main(int argc,
         char **argv) {
    const int ROWS_{4};
    /*input data*/
    Eigen::MatrixXd pts(ROWS_,
                        2);
    /*ground truth label*/
    Eigen::MatrixXd label(1,
                          1);
    /*模拟训练数据准备*/
    std::vector<double> pts_val{0., 1., 1., 0., 0., 1., 0., 1.};
    std::vector<double> label_val{0.5,};
    pts = Eigen::MatrixXd::Map(&pts_val[0],
                               pts.rows(),
                               pts.cols());
    label = Eigen::MatrixXd::Map(&label_val[0],
                                 label.rows(),
                                 label.cols());

    auto input_ = std::make_shared<yrz::Variable>(yrz::Variable(ROWS_,
                                                                2,
                                                                nullptr));
    input_->set_val(pts);
    auto label_ = std::make_shared<yrz::Variable>(yrz::Variable(ROWS_,
                                                                2,
                                                                nullptr));
    label_->set_val(label);

    /*layer定义*/
    yrz::FullConnect fc1_ = yrz::FullConnect(2,
                                             1,
                                             0.0001);
    fc1_.initWeight(6);
    yrz::CrossEntropyLoss ce_loss1_ = yrz::CrossEntropyLoss();
    yrz::ReduceSum reduce_sum1_ = yrz::ReduceSum();
    yrz::MSELoss mse_loss1_ = yrz::MSELoss();

    /*graph的定义与前馈计算*/
    auto &t2_ = fc1_.forward(input_);
    auto &t5_ = ce_loss1_.forward(t4_,
                                  label_);
    /*反向BFS遍历node*/
    std::queue<std::shared_ptr<yrz::Variable>> que;
    que.emplace(t5_);
    while (!que.empty()) {
        const auto &node = que.front();
        que.pop();
        node->_gen_by->backward(node);
        for_each(node->predecessors.begin(),
                 node->predecessors.end(),
                 [&que](const std::shared_ptr<yrz::Variable> &pred) { que.emplace(pred); });
    }
    return 0;
}