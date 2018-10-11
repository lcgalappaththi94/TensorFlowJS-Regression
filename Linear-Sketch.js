var x_vals = [];
var y_vals = [];
var m, b, w, h, prevm, prevb;
var resultSet = null;
var drawn = false;

var xDivisionFactor, yDivisionFactor;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function fileSelectHandler() {
  var oFile = $('#file_input')[0].files[0];
  parseCSV(oFile, ',', function (result) {
    resultSet = result;
  });
}

function parseCSV(file, delimiter, callback) {
  var reader = new FileReader();
  reader.onload = function () {
    var lines = this.result.split('\n');
    var result = lines.map(function (line) {
      return line.split(delimiter);
    });
    callback(result);
  }
  reader.readAsText(file);
}

function setup() {
  w = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
  h = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
  createCanvas(w, h);
  background(0);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  prevm = m;
  prevb = b;
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = mx + b;
  const ys = xs.mul(m).add(b);
  return ys;
}

function Maximum(input, col) {
  let max = Number(input[0][col]);
  for (let index = 0; index < input.length - 1; index++) {
    if (max < Number(input[index][col])) {
      max = Number(input[index][col]);
    }
  }
  return max;
}

function getDivisionFactor(parameter, max) {
  var factor;
  for (factor = 1; factor < max; factor++) {
    if (parameter >= max / factor) {
      break;
    }
  }
  return factor;
}

function drawPoints() {
  var xMax = Maximum(resultSet, 2);
  var yMax = Maximum(resultSet, 1);

  if (xMax <= w) {
    xDivisionFactor = 1;
  } else {
    xDivisionFactor = getDivisionFactor(w, xMax);
  }

  if (yMax <= h) {
    yDivisionFactor = 1;
  } else {
    yDivisionFactor = getDivisionFactor(h, yMax);
  }

  console.log("X devision Factor: " + xDivisionFactor);
  console.log("Y devision Factor: " + yDivisionFactor);

  console.log("X Max: " + xMax);
  console.log("Y Max: " + yMax);

  for (let index = 0; index < resultSet.length - 1; index++) {
    var x = Number(resultSet[index][2]) / xDivisionFactor;   //x=gdp
    var y = Number(resultSet[index][1]) / yDivisionFactor;   //y=martality
    var mappedX = map(x, 0, w, 0, 1);
    var mappedY = map(y, 0, h, 0, 1);
    x_vals.push(mappedX);
    y_vals.push(mappedY);
    //point(map(mappedX, 0, 1, 0, w), map(mappedY, 0, 1, h, 0));
  }
  drawn = true;
}

// function mousePressed() {
//   let x = map(mouseX, 0, w, 0, 1);
//   let y = map(mouseY, 0, h, 1, 0);
//   x_vals.push(x);
//   y_vals.push(y);
// }

function draw() {
  if (resultSet != null && drawn == false) {
    drawPoints();
  }

  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  background(0);
  stroke(255);
  strokeWeight(4);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, w);
    let py = map(y_vals[i], 0, 1, h, 0);
    point(px, py);
  }

  if (x_vals.length > 0) {
    const lineX = [0, 1];
    const ys = tf.tidy(() => predict(lineX));
    let lineY = ys.dataSync();
    ys.dispose();

    let x1 = map(lineX[0], 0, 1, 0, w);
    let x2 = map(lineX[1], 0, 1, 0, w);

    let y1 = map(lineY[0], 0, 1, h, 0);
    let y2 = map(lineY[1], 0, 1, h, 0);
    strokeWeight(2);

    line(x1, y1, x2, y2);
  }

  if (x_vals.length > 1) {
    if (prevm.equal(m).logicalAnd(prevb.equal(b)).dataSync() == '1') {
      console.log(m.dataSync() + " " + b.dataSync() + " ");
    }
    prevm = m;
    prevb = b;
  }
  //console.log(tf.memory().numTensors);
  //noLoop();
}