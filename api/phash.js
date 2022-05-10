import axios from "axios";
import { PrepareLobe, isLambda } from "tfjs-node-lambda-helpers";

console.log(process.versions);

const baseUrl = isLambda()
  ? `https://${process.env.VERCEL_URL}`
  : `http://localhost:3000`;
const prepareLobe = PrepareLobe(`${baseUrl}/static/model`);

let model;

export default async (req, res) => {
  const lobe = await prepareLobe.next();
  if (!lobe.done) {
    return res.status(lobe.value.statusCode).json(lobe.value);
  } else {
    model || (model = lobe.value);
  }
  const imageUrl = encodeURI(decodeURIComponent(req.query.url));

  const response = await axios.get(imageUrl, {
    responseType: "arraybuffer",
  });
  const results = model.predict(response.data);

  return res.status(200).json({ results: results.Confidences });
};
