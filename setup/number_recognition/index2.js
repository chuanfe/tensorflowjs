import * as tf from '@tensorflow/tfjs';
import $ from 'jquery';
import Chart from 'chart.js';
require('babel-polyfill');
const MnistData = require('./mnist').MnistData;

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
  
    const fillter = data.xs.cast("bool").logicalNot().cast("int32");
    const f_result = tf.where(wrong_prediction, fillter, data.xs);
    
    for (let i = 0; i < size; i++) {
      const image = f_result.slice([i, 0], [1, f_result.shape[1]]);
      const canvas = document.createElement('canvas');
      chart.appendChild(canvas);
      draw(image, canvas); 
    }
  }
  
  function logistic_regression(train_data) {
    const batch_size = parseInt($("#batch_size_input").val());
    const numIterations = parseInt($("#iteration_number_input").val());
    const number_of_labels = 10;
    let loss_results = [];
    
    const learningRate = 0.15;
    const optimizer = tf.train.sgd(learningRate);
    
    const w = tf.variable(tf.zeros([784,number_of_labels]));
    const b = tf.variable(tf.zeros([number_of_labels]));
    
    function predict(x) {
      return tf.softmax(tf.add(tf.matMul(x, w),b));
    }
    function loss(predictions, labels) {
      const entropy = tf.mean(tf.sub(tf.scalar(1),tf.sum(tf.mul(labels, tf.log(predictions)),1)));
      return entropy;
    }
    
    for (let iter = 0; iter < numIterations; iter++) {
      const batch = train_data.nextTrainBatch(batch_size);
      const train_x = batch.xs;
      const train_y = batch.labels;
      optimizer.minimize(() => {
        const loss_var = loss(predict(train_x), train_y);
        loss_results.push({
            x:new Date().getTime(), 
            y:loss_var.dataSync()[0]
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
  
  $(function() {
    let data;
    const ctx = document.getElementById("loss_chart").getContext("2d");
    const loss_chart = new Chart(ctx, {
            type: 'line',
            data: {
              labels: [],
              datasets:[{
                label: 'Loss',
                data: []
                }
              ]
            },
            options: {
              responsive: true,
              scales: {
                xAxes: [{
                  display: false,
                  scaleLabel: {
                    display: true,
                    labelString: 'Interations'
                  }
                }],
                yAxes: [{
                  display: true,
                  scaleLabel: {
                    display: true,
                    labelString: 'Loss'
                  }
                }]
              }
            }
        });
    
    async function load() {
      data = new MnistData();
      await data.load();
    }
  
    async function mnist() {
      await load();
      console.log("Data loaded!");
    }
    mnist().then(function(){
      $("#train_btn").attr("disabled",false);
      $("#train_btn").click(function() {
        const train_result = logistic_regression(data);    
   
        // Caculate accuracy
        const test_batch = data.nextTestBatch(100);
        const prediction = train_result.model(test_batch.xs);
        const correct_prediction = tf.equal(tf.argMax(prediction,1), tf.argMax(test_batch.labels, 1));
        const accuracy = tf.mean(tf.cast(correct_prediction,'float32'));
        //accuracy.print();
        // Update Loss Chart
        const loss_data = {
              labels: train_result.loss.map(d => d.x),
              datasets:[{
                label: 'Loss : accuracy'+ Math.round(accuracy.dataSync()*100)/ 100,
                backgroundColor: "rgba(159,170,174,0.3)",
                borderColor: 'rgb(201, 203, 207)',
                pointRadius: 2,
                            pointHoverRadius: 5,
                data: train_result.loss.map(d => d.y)
                }
              ]
            };
        show_result(test_batch, prediction, 100);
        loss_chart.data = loss_data;
        loss_chart.update();
      });
    }); 
  })