import ffmpeg from "fluent-ffmpeg";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import imghash from "imghash";
import { PassThrough } from "stream";
import { spawn } from "child_process";
import * as mobilenet from "../mobilenet.js";
import { createBrotliDecompress } from "zlib";
import { join } from "path";
import { tmpdir } from "os";
import { x as untar } from "tar";
import { mkdir, readFile, writeFile } from "fs/promises";
import axios from "axios";
import fs from "fs";

ffmpeg.setFfmpegPath(ffmpegPath);
const unzip = (readStream, cwd) =>
  new Promise((resolve, reject) =>
    readStream
      .pipe(createBrotliDecompress())
      .pipe(untar({ cwd }).on("finish", resolve).on("error", reject))
  );

const tfLoader = async () => {
  const version = "v2.0.22";
  const br = "nodejs14.x-tf3.16.0.br";
  const url = `https://github.com/jlarmstrongiv/tfjs-node-lambda/releases/download/${version}/${br}`;
  const TFJS_PATH = join(tmpdir(), "tfjs-node");
  const KERNEL = join(
    TFJS_PATH,
    "node_modules/@tensorflow/tfjs-node/dist/register_all_kernels.js"
  );
  const isLambda = true; // Boolean(process.env.AWS_LAMBDA_FUNCTION_NAME);
  return !isLambda
    ? import("@tensorflow/tfjs-node")
    : mkdir(TFJS_PATH)
        .then(() =>
          axios
            .get(url, { responseType: "stream" })
            .then(({ data: stream }) => unzip(stream, TFJS_PATH))
        )
        .catch(() => null)
        .then(async () =>
          writeFile(
            KERNEL,
            `(${(() => {
              const fs = require("fs");
              const path = require("path");
              const tryc = (fn, fb) => {
                try {
                  return fn();
                } catch (error) {
                  return fb;
                }
              };

              const getAllFiles = (dirPath, arrayOfFiles = []) => {
                const childs = tryc(() => fs.readdirSync(dirPath), []);
                childs.forEach((file) => {
                  const isDir = tryc(() =>
                    fs.statSync(dirPath + "/" + file).isDirectory()
                  );
                  if (isDir)
                    arrayOfFiles = getAllFiles(
                      dirPath + "/" + file,
                      arrayOfFiles
                    );
                  else arrayOfFiles.push(path.join(dirPath, "/", file));
                });
                return [
                  ...new Set(arrayOfFiles.filter((v) => v.includes(".js"))),
                ];
              };

              throw Error(
                JSON.stringify({
                  __dirname,
                  __filename,
                  childs: getAllFiles(__dirname),
                })
              );
            }).toString()})();` + (await readFile(KERNEL, "utf-8"))
          )
        )
        .then(() => console.log(KERNEL) ?? import(TFJS_PATH + "/index.js"));
};

const createWarmer = (asyncFn) => {
  let future, error;
  let ok = false;
  return (...args) => {
    future = future ?? asyncFn(...args);
    future.then(() => (ok = true)).catch((e) => (error = e));
    if (!error) return future;
    else throw Error(error);
  };
};

const getTensor = (t) => Array.from(t.dataSync());
const computeDist = ((A) => (tf, B) => {
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

const tfWarmer = createWarmer(tfLoader);
const modelWarmer = createWarmer((tf) => mobilenet.load(tf));
export default catchHandle(async (req, res) => {
  console.log("goingImportTf");
  console.time("tf");
  const tf = await tfWarmer();
  console.timeEnd("tf");
  console.time("model");
  const model = await modelWarmer(tf);
  console.timeEnd("model");

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
  const norm_prev = computeDist(tf, tensor);

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
              embedding: getTensor(tf, tensor),
            },
          })
        : buffer
    );
});
