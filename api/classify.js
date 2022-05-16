import { Transform } from "stream";
import {
  ExtractFrames,
  ffmpeg,
  isMain,
  getKnn,
  testHandler,
} from "./helpers/utils.js";
import videoTile from "./helpers/videoTile.js";
import * as movieNet from "./helpers/movieNet.js";
import * as mobileNet from "./helpers/mobileNet.js";
import * as hashNet from "./helpers/hashNet.js";
import sharp from "sharp";

let i = 0;
class Embedder extends Transform {
  constructor(getModel) {
    super();
    this.model = getModel();
    this.outputs = null;
  }

  async _transform(chunk, _, next) {
    this.outputs = await (await this.model).infer(chunk);
    next();
  }

  _flush(end) {
    this.push(JSON.stringify(this.outputs));
    end();
  }
}

const run = async (req, res) => {
  const { url: raw, model, tile, t = 2 } = req.query;
  if (!raw) throw Error("Missing video url");
  const { default: getModel, getModelInfo } = { hashNet, mobileNet, movieNet }[
    model ?? "mobileNet"
  ];
  const i = encodeURI(decodeURIComponent(raw));
  const s = getModelInfo().shape.slice(0, 2).join("x");
  (tile
    ? await videoTile(i)
    : ffmpeg({ i, s, r: 2, t: +t, vcodec: "mjpeg", f: "rawvideo" }, "pipe:")
  )
    .pipe(new ExtractFrames())
    .pipe(new Embedder(getModel))
    .pipe(
      res.writeHead(200, {
        "Content-Type": "application/json",
        "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
      }),
      { end: true }
    );
};

export default run;

if (await isMain(import.meta.url)) {
  const urls = [
    "https://dawcqwjlx34ah.cloudfront.net/86042406-c9dc-4169-b97e-7af24edf2837_1gAmPQJ5f-A.mp4",
    "https://indes-galantes-assets.s3-eu-west-1.amazonaws.com/capture104les-films-pelleas.jpg",
    "https://movies-assets.s3-eu-west-1.amazonaws.com/2c1d37b2-2e88-4d12-bbce-83972b60577f/4.jpg",
  ];
  const predicts = await Promise.all(
    urls
      .map((url) => ({ url, model: "mobileNet", tile: true }))
      .map(testHandler(run))
  ).then((arr) => arr.map((o, i) => ({ ...o, url: urls[i] })));
  console.log(getKnn(predicts, ({ output }) => output));
}
