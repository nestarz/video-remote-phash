import sharp from "sharp";
import { Transform } from "stream";
import { ExtractFrames, ffmpeg } from "./utils.js";
import movieNet, { getModelInfo as getMovieNetInfo } from "./movieNet.js";

class Embedder extends Transform {
  constructor(getModel) {
    super();
    this.predict = getModel();
    this.outputs = null;
  }

  async _transform(chunk, _, next) {
    await sharp(chunk).toFile("api/current.png");
    this.outputs = await this.predict(chunk);
    next();
  }

  _flush(end) {
    this.push(JSON.stringify(this.outputs));
    end();
  }
}

const run = (req, res) => {
  const { url: raw } = req.query;
  if (!raw) throw Error("Missing video url");
  const i = encodeURI(decodeURIComponent(raw));
  const s = getMovieNetInfo().shape.slice(0, 2).join("x");
  ffmpeg({ ss: 1, i, s, r: 2, t: 5, vcodec: "mjpeg", f: "rawvideo" }, "pipe:")
    .pipe(new ExtractFrames())
    .pipe(new Embedder(movieNet))
    .pipe(res);
};

export default run;

const DEMO_URL =
  "https://dawcqwjlx34ah.cloudfront.net/86042406-c9dc-4169-b97e-7af24edf2837_1gAmPQJ5f-A.mp4";
run({ query: { url: DEMO_URL } }, process.stdout);
