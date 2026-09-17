激活函数:为什么需要？

在数学里，线性模型（一次函数）能解决的问题极其有限。
而增加非线性激活函数，实际上是做了一个**空间映射**：它把原始数据所在的空间，映射到了一个更高维、可以被“扭曲”的特征空间。在这个新空间里，原本纠缠在一起的复杂问题，可能就变得可以被“线性可分”

类似的简单例子   阈值函数（阶跃函数Step Function）：当大于某个阈值为1，小于某个阈值为0，这样就对原始数据进行了一次分类

激活函数的作用也是如此，把线性数据映射到高维空间（并非物理上的高维，而是数学中的高维）

常见的激活函数<img width="1347" height="784" alt="image-20260917151022576" src="https://github.com/user-attachments/assets/6beefb07-2c27-4620-b4d8-1800727b2732" />

前向传播——计算损失——反向传播——梯度下降与参数更新

前向传播：用当前的参数，算出预测值

计算损失：算出损失，可以用均方误差（MSE）

反向传播：*W*(权重) →r(线性组合) → a(激活值) → y(预测值) →*L*(损失)

求∂*L*/∂*W*<img width="398" height="67" alt="image-20260917154021599" src="https://github.com/user-attachments/assets/a943974b-91d9-4ea5-bc26-f3b563d0c4fd" />


梯度下降与参数更新：
<img width="446" height="51" alt="image-20260917155716027" src="https://github.com/user-attachments/assets/322651a5-d62f-4f1e-be9a-0fe390f6fc90" />


循环往复:repeat:

PyTorch代码:

```python
for epoch in range(10000):        # 循环一万次（反复训练）
# ===== 1. 前向传播 =====
y_pred = model(x)             # 算出预测值 y_pred=28，并保存中间变量 a, r

# ===== 2. 计算损失 =====
loss = loss_fn(y_pred, y_true)# 算出 L=162

# ===== 3. 反向传播 =====
optimizer.zero_grad()         # 清空上一轮的梯度（防止累加）
loss.backward()               # PyTorch 沿着计算图自动执行链式法则！
                              # 自动填满所有参数: W.grad=144, b.grad=72, c.grad=126, d.grad=18

# ===== 4. 梯度下降与参数更新 =====
optimizer.step()              # PyTorch 执行: W = W - lr * W.grad
                              # 自动把 W 从 3 改成 1.56，b 从 1 改成 0.28...
                              
///  关于θ,g的解释。θ是所有参数的“集合”，本身是一个巨大的向量。类比封装函数，这里是封装了参数。                    
# 1. θ 已经存在于 model.parameters() 里了（打包完毕）
# 2. 算梯度：反向传播
loss.backward() 
# 这一行执行后，PyTorch 会把算出来的所有偏导数，存进每一个参数自己的 .grad 属性里。
# 此时，如果你把所有参数的 .grad 拿出来拼在一起，它就是 g！

# 3. 更新参数
optimizer.step()
# 优化器遍历每一个参数 θ_i，执行 θ_i = θ_i - lr * θ_i.grad
# 于是，新的 θ 诞生了！
```

批次

更新 

回合

