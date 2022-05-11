import { Interpreter } from "node-tflite";
import { resolve } from "path";
import fetch from "node-fetch";
import sharp from "sharp";
import { readFile } from "fs/promises";

const getModel = async (modelPath) => {
  const interpreter = new Interpreter(await readFile(modelPath));

  return {
    infer: async (buffer) => {
      interpreter.allocateTensors();
      interpreter.inputs[0].copyFrom(new Float32Array(buffer));
      interpreter.invoke();
      const outputData = new Float32Array(1001);
      interpreter.outputs[0].copyTo(outputData);
      return Array.from(outputData);
    },
  };
};

const run = async (req, res) => {
  const { url: raw } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(raw));
  const model = await getModel(resolve("static/mobilenet_v2_1.0_224.tflite"));
  const buffer = await fetch(url).then(({ body }) =>
    body.pipe(sharp()).resize(224, 224).raw({ depth: "char" }).toBuffer()
  );
  const pred = await model.infer(buffer);
  res.end(JSON.stringify(pred));
};

export default run;