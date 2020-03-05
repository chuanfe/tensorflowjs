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
    const train_y = tf.tensor1d(train_label, 'int32');
    function predict(x) {
        return tf.softmax(tf.add(tf.matMul(x, w), b));
    }

    function loss(predictions, labels) {
        const y = tf.oneHot(labels, number_of_labels);
        const entropy = tf.mean(tf.sub(tf.scalar(1), tf.sum(tf.mul(y, tf.log(predictions)), 1)));
        return entropy;
    }

    for (let iter = 0; iter < numIterations; iter++) {
        optimizer.minimize(() => {
            const loss_var = loss(predict(train_x), train_y);
            loss_var.print();
            return loss_var;
        })
    }

    return function(x) {
        const d = tf.tensor2d(x);
        var predict_result = predict(d);
        return predict_result.argMax(1).dataSync();
    }
}

// 训练结果
const regression_model = logistic_regression([[1,2],[1,3],[2,3],[2,4],[2,1],[3,2],[3,1],[4,3]],[1,1,1,1,0,0,0,0]);

console.log('[3,4],[6,7],[4,3],[7,6] 预测结果：', regression_model([[3,4],[6,7],[4,3],[7,6]]));