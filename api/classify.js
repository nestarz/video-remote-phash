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
    "https://dawcqwjlx34ah.cloudfront.net/dcc7ba87-b10f-4a21-9352-290f433292fe_342125317282510.mp4",
  ];
  const predicts = await Promise.all(
    urls
      .map((url) => ({ url, model: "mobileNet", tile: true }))
      .map(testHandler(run))
  ).then((arr) => arr.map((o, i) => ({ ...o, url: urls[i] })));
  console.log(getKnn(predicts, ({ output }) => output));
}
