import { spawn } from "child_process";
import { ffmpeg, layout, range } from "./utils.js";
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

export default async (url) => {
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

  const K = 50;
  const isVideo = duration > 1;
  const maxDuration = Math.min(duration, 80);
  const N = Math.round(Math.sqrt(K));
  const I = maxDuration / K / 1.05;
  return ffmpeg(
    {
      i: url,
      filter_complex: [
        crop && !crop.includes("-") && `crop=${crop}`,
        "crop=min(ih\\,iw):min(ih\\,iw)",
        isVideo && `scale=144:144`,
        isVideo &&
          `select='(isnan(prev_selected_t)+gte(t-prev_selected_t\,${I}))',tile=${N}x${N}`,
        false &&
          `${range(N * N)
            .map(() => "[0:v]")
            .join("")}xstack=inputs=${N * N}:layout=${layout(N)}`,
        `scale=1024:1024`,
      ]
        .filter((v) => v)
        .join(","),
      "frames:v": 1,
      vcodec: "mjpeg",
      f: "rawvideo",
    },
    "pipe:"
  ).on("error", console.error);
};
