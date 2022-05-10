import ffmpeg from "fluent-ffmpeg";
import { path as ffmpegPath } from "@ffmpeg-installer/ffmpeg";
import imghash from "imghash";
import { PassThrough } from "stream";
import { spawn } from "child_process";

ffmpeg.setFfmpegPath(ffmpegPath);

const getHash = (buffer) => imghash.hash(buffer, 8, "binary");
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

const catchHandle = (handler) => (req, res) =>
  handler(req, res).catch((error) =>
    res.writeHead(500).end(
      JSON.stringify({
        error: JSON.stringify(error, Object.getOwnPropertyNames(error)),
      })
    )
  );

export default catchHandle(async (req, res) => {
  const { url: raw, hash = "false" } = req.query;
  if (!raw) throw Error("Missing video url");
  const url = encodeURI(decodeURIComponent(raw));

  const { duration, crop } = await exec(
    ffmpegPath,
    `-i ${url} -t 1 -vf cropdetect -f null pipe:1`.split(" "),
    (res) => {
      const data = {};
      return (d) => {
        data.crop = /.*crop=(.*?)($|\n| ).*/.exec(d)?.[1] ?? data.crop;
        data.duration =
          apply(/.*Duration: (.*?),.*/.exec(d)?.[1], (v) =>
            v ? +v?.split(":")?.reduce((acc, time) => 60 * acc + +time) : v
          ) ?? data.duration;
        if (data.crop && data.duration) res(data);
      };
    }
  );

  const isVideo = duration > 0;

  const buffer = await new Promise((res) => {
    const stream = PassThrough();
    const buffers = [];
    stream.on("data", (buf) => buffers.push(buf));
    stream.on("end", () => res(Buffer.concat(buffers)));
    const N = Math.round(Math.sqrt(Math.floor(duration)));
    ffmpeg(url)
      .outputOptions([
        "-frames 1",
        `-vf ${[
          crop && !crop.includes("-") && `crop=${crop}`,
          "crop=min(ih\\,iw):min(ih\\,iw),scale=144:144",
          isVideo &&
            `select='(gte(t,1))*(isnan(prev_selected_t)+gte(t-prev_selected_t,1))',tile=${N}x${N}`,
          "scale=1024:1024"
        ]
          .filter((v) => v)
          .join(",")}`,
        "-frames:v 1",
        "-vcodec png",
        "-f rawvideo",
      ])
      .on("error", console.error)
      .pipe(stream, { end: true });
  });

  res
    .writeHead(200, {
      "Content-Type": hash === "true" ? "text/plain" : "image/png",
      "Cache-Control": `s-maxage=${86400 * 30}, stale-while-revalidate`,
    })
    .end(
      hash === "true" ? JSON.stringify({ data: await getHash(buffer) }) : buffer
    );
});
