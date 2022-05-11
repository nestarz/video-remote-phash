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
  const buffer = await fetch(url)
    .then((res) =>
      res.headers?.get("content-type")?.includes("image/")
        ? res
        : fetch(new URL(`/api/tile?url=${raw}`, `//${req.headers.host}`))
    )
    .then(({ body }) =>
      body.pipe(sharp()).resize(224, 224).raw({ depth: "char" }).toBuffer()
    );
  const tensor = await model.infer(buffer);
  res
    .writeHead(200, {
      "Content-Type": "application/json",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
    .end(JSON.stringify({ data: { tensor } }));
};

export default run;
