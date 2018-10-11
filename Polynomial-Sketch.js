
let x_vals = [];
let y_vals = [];
var w, h;
var resultSet = null;
var drawn = false;

var xDivisionFactor, yDivisionFactor;

let a, b, c, d;
let dragging = false;

const learningRate = 0.1;
const optimizer = tf.train.adam(learningRate);

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
    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(x) {
    const xs = tf.tensor1d(x);
    // y = ax^3 + bx^2 + cx + d
    const ys = xs.pow(tf.scalar(3)).mul(a)
        .add(xs.square().mul(b))
        .add(xs.mul(c))
        .add(d);
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


// function mousePressed() {
//     dragging = true;
// }

// function mouseReleased() {
//     dragging = false;
// }

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
    }
    drawn = true;
}


function draw() {

    if (resultSet != null && drawn == false) {
        drawPoints();
    }

    // if (dragging) {
    //     let x = map(mouseX, 0, width, -1, 1);
    //     let y = map(mouseY, 0, height, 1, -1);
    //     x_vals.push(x);
    //     y_vals.push(y);
    // } else {
    //     tf.tidy(() => {
    //         if (x_vals.length > 0) {
    //             const ys = tf.tensor1d(y_vals);
    //             optimizer.minimize(() => loss(predict(x_vals), ys));
    //         }
    //     });
    // }

    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
        }
    });

    // background(0);
    // stroke(255);
    // strokeWeight(4);
    // for (let i = 0; i < x_vals.length; i++) {
    //     let px = map(x_vals[i], -1, 1, 0, w);
    //     let py = map(y_vals[i], -1, 1, h, 0);
    //     point(px, py);
    // }

    background(0);
    stroke(255);
    strokeWeight(4);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], 0, 1, 0, w);
        let py = map(y_vals[i], 0, 1, h, 0);
        point(px, py);
    }


    const curveX = [];
    for (let x = -1; x <= 1; x += 0.05) {
        curveX.push(x);
    }

    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
        // let x = map(curveX[i], -1, 1, 0, w);
        // let y = map(curveY[i], -1, 1, h, 0);
        let x = map(curveX[i], 0, 1, 0, w);
        let y = map(curveY[i], 0, 1, h, 0);
        vertex(x, y);
    }
    endShape();
    console.log(a.dataSync() + " " + b.dataSync() + " " + c.dataSync() + " " + d.dataSync());

    // console.log(tf.memory().numTensors);
}