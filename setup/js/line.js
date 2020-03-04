import * as tf from '@tensorflow/tfjs';

function linear_regression(tx, ty) {
  var results = [];

  // 训练初始数据
  const train_x = tf.tensor1d(tx);
  const train_y = tf.tensor1d(ty);
  
  // 模型建立
  // const a = tf.scalar(2);
  // const b = tf.scalar(3);
  const a = tf.variable(tf.scalar(Math.random()));
  const b = tf.variable(tf.scalar(Math.random()));
  const f = x => a.mul(x).add(b);
  
  // 初始化训练参数
  const numIterations = 10;
  const learningRate = 1;
  
  // 优化器
  const optimizer = tf.train.adam(learningRate);
  console.log('optimizer', optimizer);
  // 损失函数
  const loss = (pred, label) => pred.sub(label).square().mean();
  // 训练模型
  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const loss_var = loss(f(train_x), train_y);
      loss_var.print();
      return loss_var;
    });
    const result = {};
    result.a = a.dataSync()[0];
    result.b = b.dataSync()[0];
    // result.f = x => result.a * x + result.b;
    results.push(result);
  }
  return results;
}

const regression_model = linear_regression([4,5,6,7,8,9],[12,14,16,18,20,22]);

regression_model.map((model) => {
  console.log('*****', model);
})