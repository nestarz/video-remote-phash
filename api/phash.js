import ffmpeg from "fluent-ffmpeg";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import imghash from "imghash";
import { PassThrough } from "stream";
import { spawn } from "child_process";
import * as mobilenet from "../mobilenet.js";
import { createBrotliDecompress } from "zlib";
import { pipeline } from "stream/promises";
import { createReadStream, createWriteStream } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { fetch } from "undici";
import { x as untar } from "tar";
import { mkdir } from "fs/promises";

ffmpeg.setFfmpegPath(ffmpegPath);
const unzip = (filepath, cwd) =>
  mkdir(cwd)
    .catch(() => null)
    .then(
      () =>
        new Promise((resolve, reject) =>
          createReadStream(filepath)
            .pipe(createBrotliDecompress())
            .pipe(untar({ cwd }).on("finish", resolve).on("error", reject))
        )
    );

const version = "v2.0.11";
const br = "nodejs14.x-tf2.8.6.br";
const url = `https://github.com/jlarmstrongiv/tfjs-node-lambda/releases/download/${version}/${br}`;
const filepath = join(tmpdir(), encodeURIComponent(version + br));
const TFJS_PATH = join(tmpdir(), "tfjs-node");
const isLambda = Boolean(process.env.AWS_LAMBDA_FUNCTION_NAME);
console.time("importTf");
const tf = !isLambda
  ? await import("@tensorflow/tfjs-node")
  : await pipeline((await fetch(url)).body, createWriteStream(filepath))
      .then(() => unzip(filepath, TFJS_PATH))
      .then(() => import(TFJS_PATH + "/index.js"));
console.timeEnd("importTf");
console.time("importTfModel");
const model = await mobilenet.load(tf);
console.timeEnd("importTfModel");
const getTensor = (t) => Array.from(t.dataSync());
const computeDist = ((A) => (B) => {
  const norm = A && B ? getTensor(tf.norm(tf.sub(B, A), 2, -1))[0] : null;
  A = B;
  return norm;
})();

const getHash = (buffer) => imghash.hash(buffer, 8, "binary");
const apply = (v, fn) => fn(v);
const exec = async (cmd, args, onData) => {
  const proc = spawn(cmd, args);
  return new Promise((res, rej) => {
    const data = {};
    const ok = () => res(data);
    proc.stdout.on("data", ok);
    proc.stderr.setEncoding("utf8");
    proc.stderr.on("data", onData(data, ok, rej));
    proc.on("close", ok);
  });
};

const catchHandle = (handler) => (req, res) =>
  handler(req, res).catch((error) =>
    res.writeHead(500).end(
      JSON.stringify({
        error: JSON.stringify(error, Object.getOwnPropertyNames(error)),
      })
    )
  );

export default catchHandle(async (req, res) => {
  const { url: raw, hash = "false" } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(raw));

  console.time("crop_duration");
  const { duration, crop } = await exec(
    ffmpegPath,
    [
      `-i ${url}`,
      "-t 3",
      "-vf",
      "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)',format=gbrp,tblend=all_mode=difference,curves=all='0/0 0.3/1 1/1',cropdetect",
      `-vsync vfr -f null pipe:1`,
    ].flatMap((d) => (d[0] === "-" ? d.split(" ") : d)),
    (data, res) => (s) => {
      data.crop = /.*crop=(.*?)($|\n| ).*/.exec(s)?.[1] ?? data.crop;
      data.duration =
        apply(/.*Duration:[\n\r\s]+(.*?),.*/.exec(s)?.[1], (v) =>
          v ? +v?.split(":")?.reduce((acc, time) => 60 * acc + +time) : v
        ) ?? data.duration;
      if (data.crop && data.duration) res();
    }
  );
  console.timeEnd("crop_duration");

  console.time("tile");
  const buffer = await new Promise((res) => {
    const stream = PassThrough();
    const buffers = [];
    stream.on("data", (buf) => buffers.push(buf));
    stream.on("end", () => res(Buffer.concat(buffers)));
    const isVideo = duration > 0;
    const N = Math.round(Math.sqrt(Math.floor(duration)));
    ffmpeg(url)
      .outputOptions([
        `-vf ${[
          crop && !crop.includes("-") && `crop=${crop}`,
          "crop=min(ih\\,iw):min(ih\\,iw),scale=144:144",
          isVideo &&
            `select='(isnan(prev_selected_t)+gte(t-prev_selected_t,1))',tile=${N}x${N}`,
          `scale=1024:1024`,
        ]
          .filter((v) => v)
          .join(",")}`,
        "-frames:v 1",
        "-vcodec png",
        "-f rawvideo",
      ])
      .on("error", console.error)
      .pipe(stream, { end: true });
  });
  console.timeEnd("tile");

  console.time("embedding");
  const tfimage = tf.node.decodeImage(buffer);
  const tensor = await model.infer(tfimage, true);
  console.timeEnd("embedding");
  const norm_prev = computeDist(tensor);

  res
    .writeHead(200, {
      "Content-Type": hash === "true" ? "application/json" : "image/png",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
    .end(
      hash === "true"
        ? JSON.stringify({
            data: {
              phash: await getHash(buffer),
              norm_prev,
              embedding: getTensor(tensor),
            },
          })
        : buffer
    );
});
