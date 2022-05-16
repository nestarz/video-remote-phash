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
