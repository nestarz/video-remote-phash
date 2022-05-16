import "@tensorflow/tfjs-backend-cpu";
import * as tflite from "tfjs-tflite-node";
import * as tf from "@tensorflow/tfjs-core";
import { bufferToTensor } from "./utils.js";
import { resolve } from "path";

const modelPath = resolve(
  "static",
  "lite-model_imagenet_mobilenet_v3_large_100_224_feature_vector_5_default_custom100.tflite"
);
const [H, W, C] = [224, 224, 3];
const tfliteModel = await tflite.loadTFLiteModel(modelPath);

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
        output: Array.from(output),
        length: state.length,
      };
    },
  };
};
