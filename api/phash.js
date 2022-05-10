import { Interpreter } from "node-tflite";
import fs from "fs";
import path from "path";
import { createCanvas, loadImage } from "canvas";
import axios from "axios";

const modelPath = path.resolve(
  __dirname,
  "..",
  "static",
  "mobilenet_v1_1.0_224_quant.tflite"
);

function createInterpreter() {
  const modelData = fs.readFileSync(modelPath);
  return new Interpreter(modelData);
}

async function getImageInput(arrayBuffer, size) {
  const canvas = createCanvas(size, size);
  const context = canvas.getContext("2d");
  const image = await loadImage(arrayBuffer);
  context.drawImage(image, 0, 0, size, size);
  const data = context.getImageData(0, 0, size, size);

  const inputData = new Uint8Array(size * size * 3);

  for (let i = 0; i < size * size; ++i) {
    inputData[i * 3] = data.data[i * 4];
    inputData[i * 3 + 1] = data.data[i * 4 + 1];
    inputData[i * 3 + 2] = data.data[i * 4 + 2];
  }

  return inputData;
}

export default async (req, res) => {
  const interpreter = createInterpreter();
  interpreter.allocateTensors();

  const imageUrl = encodeURI(decodeURIComponent(req.query.url));
  const { data } = await axios.get(imageUrl, {
    responseType: "arraybuffer",
  });
  const inputData = await getImageInput(data, 224);
  interpreter.inputs[0].copyFrom(inputData);

  interpreter.invoke();

  const outputData = new Uint8Array(1001);
  interpreter.outputs[0].copyTo(outputData);

  const maxIndex = outputData.indexOf(Math.max(...Array.from(outputData)));

  res.end(JSON.stringify(maxIndex));
};
