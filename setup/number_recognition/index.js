import * as tf from '@tensorflow/tfjs';
import $ from 'jquery';
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
  // 训练数据
  const chart = document.getElementById('chart');
  const batch_size = 100;
  const test_batch = data.nextTrainBatch(batch_size);
  console.log('batch', test_batch);
  for (let iter = 0; iter < batch_size; iter++) {
    const image = test_batch.xs.slice([iter, 0], [1, test_batch.xs.shape[1]]);
    const canvas = document.createElement('canvas');
    chart.appendChild(canvas);
    draw(image, canvas);
  }
  // 训练结果
  const train_result = number_recognition(data);
  console.log('####', train_result);
  const prediction = train_result.model(test_batch.xs);
  const correct_prediction = tf.equal(tf.argMax(prediction,1), tf.argMax(test_batch.labels, 1));
  const accuracy = tf.mean(tf.cast(correct_prediction,'float32'));
  console.log('预测结果', prediction, correct_prediction, accuracy,train_result.loss);
  show_result(data.nextTrainBatch(100), prediction, 100)
})

function number_recognition(train_data) {
  const batch_size = 3;
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

// 绘制canvas
function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function show_result(data, predict, size) {
  $('#result').empty();
  const chart = document.getElementById('result');
  const labels = data.labels;
  const wrong_prediction = tf.notEqual(tf.argMax(predict,1), tf.argMax(labels, 1));
  console.log('^^^^^^^^^^wrong_prediction', wrong_prediction)

  const fillter = data.xs.cast("bool").logicalNot().cast("int32");
  console.log('fillter', fillter, data.xs)
  const f_result = tf.where(wrong_prediction, fillter, data.xs);
  console.log('22222222222', f_result)
  
  for (let i = 0; i < size; i++) {
    const image = f_result.slice([i, 0], [1, f_result.shape[1]]);
    const canvas = document.createElement('canvas');
    chart.appendChild(canvas);
    draw(image, canvas); 
  }
}