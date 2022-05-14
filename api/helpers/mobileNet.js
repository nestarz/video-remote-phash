import { resolve } from "path";
import "@tensorflow/tfjs-backend-cpu";
import * as tflite from "tfjs-tflite-node";
import * as tf from "@tensorflow/tfjs-core";
import { bufferToTensor } from "./utils.js";
import { readFile } from "fs/promises";

const modelPath = resolve("static/mobilenet_v2_1.0_224.tflite");
const model = resolve(modelPath);
const [H, W, C] = [224, 224, 3];
const tfliteModel = await tflite.loadTFLiteModel(model);

const labelPath = "static/labels_mobilenet_quant_v1_224.txt";
const argmax = (arr) => arr.reduce((m, x, i, arr) => (x > arr[m] ? i : m), 0);
const labels = await readFile(labelPath, "utf-8").then((a) => a.split("\n"));

export const getModelInfo = () => ({ shape: [H, W, C] });

export default async () => {
  return {
    infer: async (buffer) => {
      const tensor = await bufferToTensor(buffer, H, W, 1);
      const logits = tfliteModel.predict(tensor);
      const output = Array.from(await tf.mul(tf.add(logits, 1), 127.5).data());
      return { output, labels: labels[argmax(output)] };
    },
  };
};
