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

const range = (k) => [...Array(k).keys()];
const isString = (o) => typeof o === "string";
const toParams = (o) =>
  o.flatMap((p) =>
    isString(p) ? p : Object.entries(p).flatMap(([k, v]) => [`-${k}`, v])
  );
const logCmd = (a, b) => console.log(a, b.join(" ")) ?? [a, b];
export const ffmpeg = (...o) =>
  spawn(...logCmd(ffmpegPath, toParams(o))).stdout;

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
}
