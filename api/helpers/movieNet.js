import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs-core";
import * as tflite from "tfjs-tflite-node";
import { readFile } from "fs/promises";
import { bufferToTensor } from "./utils.js";
import { resolve } from "path";

const modelPath = resolve(
  "static/lite-model_movinet_a0_stream_kinetics-600_classification_tflite_float16_2.tflite"
);
const labelPath = resolve("static/kinetics-i3d_label_map_600.txt");
const [H, W, C] = [172, 172, 3];

const argmax = (arr) => arr.reduce((m, x, i, arr) => (x > arr[m] ? i : m), 0);
const find = (d, obj) => Object.entries(obj).find(([k]) => d.includes(k))[1];

const labels = await readFile(labelPath, "utf-8").then((a) => a.split("\n"));
const model = await tflite.loadTFLiteModel(modelPath);

const quantizedScale = ({ name, dtype, shape, quantization: [s, zP] }, state) =>
  name?.includes("frame_count") || dtype === "float32" || scale === 0
    ? state ?? tf.zeros(shape, dtype)
    : tf.cast(tf.sum(tf.div(state, s), zP), dtype);

const getSlug = (name) => name.slice("serving_default_".length, -":0".length);
const inputs = model.modelRunner
  .getInputs()
  .map((input) => ({ input }))
  .map(({ input, input: { name, shape, dataType: d, quantization } }, i) => ({
    index: i,
    input,
    shape: shape.split(",").map((d) => +d),
    quantization: quantization?.split(",")?.map((d) => +d) ?? [1, 0],
    dtype: find(d, { int: "int32", float: "float32", bool: "bool" }),
    name,
    slug: getSlug(name),
  }))
  .map((i) => ({ ...i, state: quantizedScale(i) }));

const getOutput = (obj) => {
  const outputs = Object.values(obj);
  const predIndex = outputs.findIndex((t) => t.size === labels.length);
  const getName = (i) => inputs.filter((v) => v.slug !== "image")[i].name;
  return [
    outputs[predIndex],
    outputs
      .filter((_, i) => i !== predIndex)
      .reduce((p, v, i) => ({ ...p, [getName(i)]: v }), {}),
  ];
};

export const getModelInfo = () => ({ shape: [H, W, C] });

export default async () => {
  let states = inputs.reduce((p, o) => ({ ...p, [o.name]: o.state }), {});
  return {
    infer: async (buffer) => {
      const clip = await bufferToTensor(buffer, H, W, 2);
      const { name, ...o } = inputs.find(({ slug }) => slug === "image");
      const [logits, newStates] = getOutput(
        model.predict({ ...states, [name]: quantizedScale(o, clip) })
      );
      states = newStates;
      const output = Array.from(await tf.mul(tf.add(logits, 1), 127.5).data());
      return { output, labels: labels[argmax(output)] };
    },
  };
};
