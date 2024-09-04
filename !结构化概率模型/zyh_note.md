# 15. 表示学习 + 16. 结构化概率模型

- 任务：
  - 15+16：**ddl:10/10**
    - [ ] 更新Pytorch代码
    - [ ] 优化和更新本章内容
      - [ ] 核对原书中文翻译
      - [ ] 撰写笔记
    - [ ] 提交Pull Request
  - 其他可优化的地方：
    - 可将笔记合并为tex格式，以便生成电子书
    - 修正其他章节内容和.md格式、排版问题

## 15. 表示学习

$$ \nabla_cL=\sum\limits_t {\left( \frac{\partial o^{(t)}}{\partial c}\right)}^{T} \nabla_{o^{(t)}}L =\sum\limits_t \nabla_{o^{(t)}}L \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $$ $$ \nabla_bL=\sum\limits_t {\left( \frac{\partial h^{(t)}}{\partial b^{(t)}}\right)}^{T} \nabla_{h^{(t)}}L =\sum\limits_t diag\left( 1- (h^{(t)})^2\right) \nabla_{h^{(t)}}L $$ $$ \nabla_VL=\sum\limits_t\sum\limits_i {\left( \frac{\partial L}{\partial o_i^{(t)}}\right)}^{T} \nabla_{V}L =\nabla_Vo_i^{(t)} =\sum\limits_t (\nabla_{o^{(t)}}L)h^{(t)^T} $$ $$ \nabla_WL=\sum\limits_t\sum\limits_i {\left( \frac{\partial L}{\partial h_i^{(t)}}\right)}^{T} \nabla_{W^{(t)}}h_i^{(t)} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $$ $$ =\sum\limits_t diag\left( 1- (h^{(t)})^2\right) (\nabla_{h^{(t)}}L) h^{(t-1)^T} \ \ \ \ \ \ \ \ \ \ \ \ $$ $$ \nabla_UL=\sum\limits_t\sum\limits_i {\left( \frac{\partial L}{\partial h_i^{(t)}}\right)}^{T} \nabla_{U^{(t)}}h_i^{(t)} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $$ $$ =\sum\limits_t diag\left( 1- (h^{(t)})^2\right) (\nabla_{h^{(t)}}L) x^{(t)^T} \ \ \ \ \ \ \ \ \ \ \ \ \ \ $$ 除了梯度表达式不同，RNN的反向传播算法和DNN区别不大。