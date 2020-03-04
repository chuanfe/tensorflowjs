import * as tf from '@tensorflow/tfjs';

function logistic_regression(train_data, train_label) {
    console.log(train_data, train_label)
    var results = [];
    const numIterations = 100;
    const learningRate = 0.1;
    // 优化器
    const optimizer = tf.train.adam(learningRate);

    const number_of_labels = Array.from(new Set(train_label)).length;
    const number_of_data = train_label.length;

    // 模型预测数据
    const w = tf.variable(tf.zeros([2, number_of_labels]));
    const b = tf.variable(tf.zeros([number_of_labels]));

    // 训练初始数据
    const train_x = tf.tensor2d(train_data);
    const train_y = tf.tensor1d(train_label);

    function predict(x) {
        return tf.softmax(tf.add(tf.matMul(x, w), b));
    }

    function loss(predictions, labels) {
        const y = tf.oneHot(labels, number_of_labels);
        // const entropy = tf.mean(tf.sub(tf.scalar(1), tf.sum((tf.mul(y, tf.log(predictions)))))
        const entropy = tf.mean(tf.sub(tf.scalar(1), tf.sum(tf.mul(y, tf.log(predictions)), 1)));
        return entropy;
    }

    for (let iter = 0; iter < numIterations; iter++) {
        optimizer.minimize(() => {
            const loss_var = loss(predict(train_x), train_y);
            loss_var.print();
            return loss_var;
        })
        // const result = {};
        // result.a = w.dataSync()[0];
        // result.b = b.dataSync()[0];
        // // result.f = x => result.a * x + result.b;
        // results.push(result);
    }

    return function(x) {
        const d = tf.tensor2d(x);
        var predict_result = predict(d);
        return predict_result.argMax(1).dataSync();
    }
}

const regression_model = logistic_regression([[1,2],[3,4],[5,6],[6,7]],[0,0,1,1]);

// regression_model.map((model) => {
//   console.log('@@@@', model);
// })