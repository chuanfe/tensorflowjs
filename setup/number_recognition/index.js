import * as tf from '@tensorflow/tfjs';
require('babel-polyfill');
const MnistData = require('./mnist').MnistData;

let data;

async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  await load();
  console.log("Data loaded!");
}

mnist().then(function(){
  console.log('*****', data);
  const train_result = number_recognition(data);
  console.log('####', train_result);
})

function number_recognition(train_data) {
  const batch_size = 10;
  const numIterations = 100;
  const number_of_labels = 10;
  let loss_results = [];

  // 优化器
  const learningRate = 0.15;
  const optimizer = tf.train.sgd(learningRate);

  const w = tf.variable(tf.zeros([784, number_of_labels]));
  const b = tf.variable(tf.zeros([number_of_labels]));

  function predict(x) {
    return tf.softmax(tf.add(tf.matMul(x, w), b));
  }

  function loss(predictions, labels) {
    const entropy = tf.mean(tf.sub(tf.scalar(1), tf.sum(tf.mul(labels, tf.log(predictions)), 1)));
    return entropy;
  }

  for (let iter = 0; iter < numIterations; iter++) {
    const batch = train_data.nextTrainBatch(batch_size);
    const train_x = batch.xs;
    const train_y = batch.labels;
    optimizer.minimize(() => {
      const loss_var = loss(predict(train_x), train_y);
      loss_results.push({
        x: new Date().getTime(),
        y: loss_var.dataSync()[0]
      });
      return loss_var;
    })
    train_x.dispose();
    train_y.dispose();
  }

  return { 
    model : predict,
    loss : loss_results
  };
}