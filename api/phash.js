import ffmpeg from "fluent-ffmpeg";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import imghash from "imghash";
import { PassThrough } from "stream";
import { spawn } from "child_process";

ffmpeg.setFfmpegPath(ffmpegPath);

const apply = (v, fn) => fn(v);
const exec = async (cmd, args, onData) => {
  const proc = spawn(cmd, args);
  return new Promise((res, rej) => {
    proc.stdout.on("data", res);
    proc.stderr.setEncoding("utf8");
    proc.stderr.on("data", onData(res, rej));
    proc.on("close", res);
  });
};

export default async (req, res) => {
  const { url, hash = "false" } = req.query;
  if (!url) throw Error("Missing video url");

  const crop = await exec(
    ffmpegPath,
    `-i ${url} -t 1 -vf cropdetect -f null pipe:1`.split(" "),
    (res) => (d) =>
      apply(/.*crop=(.*?)($|\n| ).*/.exec(d), (v) => (v ? res(v[1]) : null))
  );

  const buffer = await new Promise((res) => {
    const stream = PassThrough();
    const buffers = [];
    stream.on("data", (buf) => buffers.push(buf));
    stream.on("end", () => res(Buffer.concat(buffers)));
    ffmpeg(url)
      .outputOptions([
        "-frames 1",
        `-vf crop=${crop},crop=min(ih\\,iw):min(ih\\,iw),scale=50:50,select='(gte(t\,2))*(isnan(prev_selected_t)+gte(t-prev_selected_t\,2))',tile=5x5`,
        "-frames:v 1",
        "-vcodec png",
        "-f rawvideo",
      ])
      .on("error", () => null)
      .pipe(stream, { end: true });
  });

  res
    .writeHead(200, {
      "Content-Type": hash === "true" ? "text/plain" : "image/png",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
    .end(hash === "true" ? await imghash.hash(buffer) : buffer);
};
