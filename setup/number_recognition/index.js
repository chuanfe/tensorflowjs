require('babel-polyfill')
const MnistData = require('./mnist').MnistData;

let data;
const batch_size = 10;
const chart = document.getElementById('chart');
async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  await load();
  console.log("Data loaded!");
}
mnist().then(function () {
  const batch = data.nextTrainBatch(batch_size);
  // 使用train_data数据
  console.log('batch', batch);
  for (let iter = 0; iter < batch_size; iter++) {
    const image = batch.xs.slice([iter, 0], [1, batch.xs.shape[1]]);
    const canvas = document.createElement('canvas');
    chart.appendChild(canvas);
    draw(image, canvas);
  }
});

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