import { resolve } from "path";
import fetch from "node-fetch";
import sharp from "sharp";
import { fileTypeFromStream } from "file-type";
import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs-core";
import * as tflite from "tfjs-tflite-node";

const getModel = async (modelPath) => {
  const model = resolve(modelPath);
  const tfliteModel = await tflite.loadTFLiteModel(model);

  return {
    infer: async (buffer) => {
      const tensor = tf.reshape(
        tf.tensor(new Float32Array(buffer)),
        [224, 224, -1]
      );
      const input = tf.sub(tf.div(tf.expandDims(tensor), 127.5), 1);
      return tfliteModel.predict(input);
    },
  };
};

const run = async (req, res) => {
  const { url: raw } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(decodeURIComponent(raw)));
  const model = await getModel(resolve("static/mobilenet_v2_1.0_224.tflite"));
  const mime = (await fileTypeFromStream((await fetch(url)).body)).mime;
  const loc = req.headers.host.includes("0.0.");
  const tileUrl = loc ? "http://localhost:3000" : `https://${req.headers.host}`;

  const buffer = await (mime.includes("video")
    ? fetch(new URL(`/api/tile?url=${raw}`, tileUrl))
    : fetch(url)
  ).then(({ body }) =>
    body.pipe(sharp()).resize(224, 224).raw({ depth: "uchar" }).toBuffer()
  );
  const tensor = await model.infer(buffer);
  res
    .writeHead(200, {
      "Content-Type": "application/json",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
    .end(JSON.stringify({ data: { tensor: Array.from(tensor.dataSync()) } }));
};

export default run;
