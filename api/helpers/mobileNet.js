import "@tensorflow/tfjs-backend-cpu";
import * as tflite from "tfjs-tflite-node";
import * as tf from "@tensorflow/tfjs-core";
import { bufferToTensor, download } from "./utils.js";
import { readFile } from "fs/promises";
import { resolve } from "path";

const modelPath = await download(
  "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/imagenet/mobilenet_v3_large_100_224/classification/5/default/1.tflite",
  "lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_default_1.tflite"
);
const [H, W, C] = [224, 224, 3];
const tfliteModel = await tflite.loadTFLiteModel(modelPath);

const labelPath = resolve("static/labels_mobilenet_quant_v1_224.txt");
const labels = await readFile(labelPath, "utf-8").then((a) => a.split("\n"));

export const getModelInfo = () => ({ shape: [H, W, C] });

export default async () => {
  const state = [];
  return {
    infer: async (buffer) => {
      const tensor = await bufferToTensor(buffer, H, W, 1);
      state.push(tfliteModel.predict(tensor));
      const output = await tf
        .div(tf.sum(tf.stack(state), 0), state.length)
        .data();
      return {
        scores: Object.entries(output)
          .sort(([, a], [, b]) => b - a)
          .map(([k, v]) => ({ label: labels[+k], score: v }))
          .slice(0, 10),
        output: Array.from(output),
      };
    },
  };
};
