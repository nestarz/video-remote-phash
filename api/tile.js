import { createWriteStream } from "fs";
import { isMain } from "./helpers/utils.js";
import videoTile from "./helpers/videoTile.js";

const run = async (req, res) => {
  const { url: raw } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(raw));

  (await videoTile(url)).pipe(
    res.writeHead(200, {
      "Content-Type": "image/png",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
  );
};

export default run;

if (isMain(import.meta.url)) {
  const DEMO_URL =
    "https://dawcqwjlx34ah.cloudfront.net/86042406-c9dc-4169-b97e-7af24edf2837_1gAmPQJ5f-A.mp4";
  run(
    { query: { url: DEMO_URL } },
    { writeHead: () => createWriteStream("test.png") }
  );
}
