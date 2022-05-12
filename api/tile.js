import ffmpeg from "fluent-ffmpeg";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import { PassThrough } from "stream";
import { spawn } from "child_process";

ffmpeg.setFfmpegPath(ffmpegPath);

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

const range = (N) => [...Array(N).keys()];
const outer = ([v1, v2]) => v1.map((x) => v2.map((y) => `${x}_${y}`));
const tile = (N, fn) =>
  range(N).map((i, _, arr) => arr.slice(0, i).map(fn).join("+") || 0);
const layout = (N) =>
  outer([tile(N, (k) => `w${k * N}`), tile(N, (k) => `h${k}`)])
    .flat()
    .join("|");

export default async (req, res) => {
  const { url: raw } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(raw));

  console.time("crop");
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
  console.timeEnd("crop");

  console.time("ffmpeg");
  const buffer = await new Promise((res) => {
    const stream = PassThrough();
    const buffers = [];
    stream.on("data", (buf) => buffers.push(buf));
    stream.on("end", () => res(Buffer.concat(buffers)));
    const k0 = 100;
    const I = duration / k0;
    const N = Math.round(Math.sqrt(k0));
    const K = N * N;

    ffmpeg()
      .outputOptions([
        ...range(K).flatMap((k) => [
          "-noaccurate_seek",
          `-ss ${k * I}`,
          `-i ${url}`,
        ]),
        "-frames:v 1",
        `-filter_complex ${range(K)
          .map(
            (k) => `[${k}:v]crop=min(ih\\,iw):min(ih\\,iw),scale=144:144[v${k}]`
          )
          .join(";")};${range(K)
          .map((k) => `[v${k}]`)
          .join("")}xstack=inputs=${K}:layout=${layout(N)},scale=1024:1024`,
        "-vcodec png",
        "-preset ultrafast",
        "-f rawvideo",
      ])
      .on("start", console.log)
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
