import { PassThrough } from "stream";
import { spawn } from "child_process";
import { ffmpeg } from "./helpers/utils.js";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";

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

const run = async (req, res) => {
  const { url: raw } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(raw));

  console.time("ffmpeg");
  const { duration, crop } = await exec(
    ffmpegPath,
    [
      "-ss 3",
      `-i ${url}`,
      "-t 5",
      "-vf",
      "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)',format=gbrp,tblend=all_mode=difference,curves=all='0/0 0.35/1 1/1',cropdetect",
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

  const buffer = await new Promise((res) => {
    const stream = PassThrough();
    const buffers = [];
    stream.on("data", (buf) => buffers.push(buf));
    stream.on("end", () => res(Buffer.concat(buffers)));
    const isVideo = duration > 0;
    const K = 50;
    const maxDuration = Math.min(duration, 120);
    const N = Math.round(Math.sqrt(K));
    const I = maxDuration / K / 1.05;
    ffmpeg(
      {
        i: url,
        vf: [
          crop && !crop.includes("-") && `crop=${crop}`,
          "crop=min(ih\\,iw):min(ih\\,iw),scale=144:144",
          isVideo &&
            `select='(isnan(prev_selected_t)+gte(t-prev_selected_t\,${I}))',tile=${N}x${N}`,
          `scale=1024:1024`,
        ]
          .filter((v) => v)
          .join(","),
        "frames:v": 1,
        vcodec: "png",
        f: "rawvideo",
      },
      "pipe:"
    )
      .on("error", console.error)
      .pipe(stream, { end: true });
  });
  console.timeEnd("ffmpeg");

  res
    .writeHead(200, {
      "Content-Type": "image/png",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
    .end(buffer);
};

export default run;

import("url")
  .then(({ fileURLToPath: fn }) => process.argv[1] === fn(import.meta.url))
  .then((isMain) => {
    const DEMO_URL =
      "https://dawcqwjlx34ah.cloudfront.net/86042406-c9dc-4169-b97e-7af24edf2837_1gAmPQJ5f-A.mp4";
    if (isMain)
      run(
        { query: { url: DEMO_URL, model: "movsie" } },
        { writeHead: () => process.stdout }
      );
  });
