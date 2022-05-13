import { resolve } from "path";
import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs-core";
import * as tflite from "tfjs-tflite-node";
import { readFileSync } from "fs";

const model = resolve("static/mobilenet_v2_1.0_224.tflite");
const tfliteModel = await tflite.loadTFLiteModel(model);

// Prepare input tensors.
const img = tf.reshape(
  tf.tensor(new Uint8Array(readFileSync("static/img.jpg"))),
  [720, 1080, -1]
);
const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);

// Run inference and get output tensors.
let outputTensor = tfliteModel.predict(input);
console.log(outputTensor.dataSync());
