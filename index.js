import { createServer } from "http";
import { URL, parse } from "url";
import * as phash from "./api/phash.js";

const newUrl = (req) =>
  new URL(
    /^http:/.test(req.url) ? req.url : "http://" + req.headers.host + req.url
  );

const createApp = (handlers = []) => ({
  get: (url, handler) => handlers.push({ method: "GET", url, handler }),
  post: (url, handler) => handlers.push({ method: "POST", url, handler }),
  listen: (port, onListen) =>
    createServer(
      (req, res) =>
        handlers
          .filter(({ method }) => method === req.method)
          .filter(({ url }) => url === newUrl(req).pathname)
          .map(({ handler: f }) => f({ ...req, ...parse(req.url, true) }, res))
          .map((r) => Promise.resolve(r).catch(console.error)).length === 0 &&
        res.writeHead(404).end("404")
    ).listen(port, () => onListen({ port })),
});

const app = createApp();

app.get("/", phash.default);

app.listen(process.env.PORT ?? 3000, ({ port }) => {
  console.log(`Example app listening on port ${port}`);
});
