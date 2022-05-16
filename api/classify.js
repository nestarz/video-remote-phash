import sharp from "sharp";
import { Transform } from "stream";
import { ExtractFrames, ffmpeg } from "./helpers/utils.js";
import * as movieNet from "./helpers/movieNet.js";
import * as mobileNet from "./helpers/mobileNet.js";

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

const run = (req, res) => {
  const { url: raw, model, tile } = req.query;
  if (!raw) throw Error("Missing video url");
  const { default: getModel, getModelInfo } =
    model === "movie" ? movieNet : mobileNet;
  const url = encodeURI(decodeURIComponent(raw));
  const loc = !req.headers || req.headers.host.includes("0.0.");
  const tileUrl = loc ? "http://localhost:3000" : `https://${req.headers.host}`;
  const i = tile ? new URL(`/api/tile?url=${raw}`, tileUrl) : url;
  const s = getModelInfo().shape.slice(0, 2).join("x");
  ffmpeg({ i, s, r: 2, t: 5, vcodec: "mjpeg", f: "rawvideo" }, "pipe:")
    .pipe(new ExtractFrames())
    .pipe(new Embedder(getModel))
    .pipe(
      res.writeHead(200, {
        "Content-Type": "application/json",
        "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
      })
    );
};

export default run;

import("url")
  .then(({ fileURLToPath: fn }) => process.argv[1] === fn(import.meta.url))
  .then((isMain) => {
    const DEMO_URL =
      "https://www.academiedugout.fr/images/17157/1200-auto/pomme_000.jpg";
    if (isMain)
      run(
        { query: { url: DEMO_URL, model: "movie" } },
        { writeHead: () => process.stdout }
      );
  });
