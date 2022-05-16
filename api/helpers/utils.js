import { Transform } from "stream";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import { spawn } from "child_process";
import sharp from "sharp";
import * as tf from "@tensorflow/tfjs-core";
import { createWriteStream } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import fetch from "node-fetch";
import { pipeline } from "stream/promises";
import { access } from "fs/promises";

export const noopLog = (v) => console.log(v) ?? v;
export const range = (k) => [...Array(k).keys()];
const isString = (o) => typeof o === "string";
const toParams = (o) =>
  o.flatMap((p) =>
    isString(p) ? p : Object.entries(p).flatMap(([k, v]) => [`-${k}`, v])
  );
const logCmd = (a, b) => console.log(a, b.join(" ")) ?? [a, b];
export const ffmpeg = (...o) => {
  const p = spawn(...logCmd(ffmpegPath, toParams(o)));
  //p.stderr.on("data", (v) => console.log(String(v)));
  p.stderr.on("error", (v) => console.log(String(v)));
  return p.stdout;
};

const imageToTensor = ({ data, info: { height: h, width: w, channels: c } }) =>
  tf.sub(tf.div(tf.tensor3d(data, [h, w, c], "float32"), 127.5), 1);

export const bufferToTensor = (buffer, H, W, p = 0) =>
  sharp(buffer)
    .resize(H, W)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
    .then(imageToTensor)
    .then((t) => range(p).reduce((t) => tf.expandDims(t), t));

const tmpPath = (path) => join(tmpdir(), path);
export const download = (url, path) =>
  access(tmpPath(path))
    .catch(async () =>
      pipeline((await fetch(url)).body, createWriteStream(tmpPath(path)))
    )
    .then(() => tmpPath(path));

export class ExtractFrames extends Transform {
  constructor(magicNumberHex = "FFD8FF") {
    super({ readableObjectMode: true });
    this.magicNumber = Buffer.from(magicNumberHex, "hex");
    this.currentData = Buffer.alloc(0);
  }

  _transform(newData, encoding, done) {
    // Add new data
    this.currentData = Buffer.concat([this.currentData, newData]);

    // Find frames in current data
    while (true) {
      // Find the start of a frame
      const startIndex = this.currentData.indexOf(this.magicNumber);
      if (startIndex < 0) break; // start of frame not found

      // Find the start of the next frame
      const endIndex = this.currentData.indexOf(
        this.magicNumber,
        startIndex + this.magicNumber.length
      );
      if (endIndex < 0) break; // we haven't got the whole frame yet

      // Handle found frame
      this.push(this.currentData.slice(startIndex, endIndex)); // emit a frame
      this.currentData = this.currentData.slice(endIndex); // remove frame data from current data
      if (startIndex > 0)
        console.error(`Discarded ${startIndex} bytes of invalid data`);
    }
    done();
  }
  _flush(end) {
    const startIndex = this.currentData.indexOf(this.magicNumber);
    end(null, startIndex >= 0 ? this.currentData.slice(startIndex) : null);
  }
}

export const testHandler = (run) => (query) => {
  class Log extends Transform {
    constructor(fn) {
      super();
      this.fn = fn;
    }
    async _transform(chunk, _, end) {
      console.log(query, String(chunk));
      await Promise.resolve(this.fn(JSON.parse(chunk)));
      end();
    }
  }

  return new Promise((res) =>
    run({ query }, { writeHead: () => new Log(res) })
  );
};

export const isMain = (url) =>
  import("url").then(
    ({ fileURLToPath: fn }) => false && process.argv[1] === fn(url)
  );

export const norm = (a, b) =>
  a
    .map((x, i) => Math.abs(x - b[i]) ** 2) // square the difference
    .reduce((sum, now) => sum + now) ** // sum
  (1 / 2);

export const getKnn = (arr, getVec) =>
  arr.flatMap((a, i) =>
    arr
      .filter((_, j) => i < j)
      .map((b) => ({
        a: a.url,
        b: b.url,
        distance: norm(getVec(a), getVec(b)),
      }))
      .sort(({ distance: a }, { distance: b }) => a - b)
  );

const outer = ([v1, v2]) => v1.map((x) => v2.map((y) => `${x}_${y}`));
const tile = (N, fn) =>
  range(N).map((i, _, arr) => arr.slice(0, i).map(fn).join("+") || 0);
export const layout = (N) =>
  outer([tile(N, (k) => `w${k * N}`), tile(N, (k) => `h${k}`)])
    .flat()
    .join("|");
