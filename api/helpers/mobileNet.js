import "@tensorflow/tfjs-backend-cpu";
import * as tflite from "tfjs-tflite-node";
import * as tf from "@tensorflow/tfjs-core";
import { bufferToTensor, download } from "./utils.js";

const modelPath = await download(
  "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/imagenet/mobilenet_v3_large_100_224/feature_vector/5/default/1.tflite",
  "lite-model_imagenet_mobilenet_v3_large_100_224_feature_vector_5_default_4.tflite"
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
