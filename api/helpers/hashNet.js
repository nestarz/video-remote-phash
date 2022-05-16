import { imageHash } from "image-hash";

export const getModelInfo = () => ({ shape: [1024, 1024, 3] });

const fromHexString = (hexString) =>
  BigInt("0x" + hexString)
    .toString(2)
    .padStart(Buffer.from(hexString, "hex").length * 8, "0")
    .split("")
    .map((d) => +d);

const phash = (data) =>
  new Promise((res, rej) =>
    imageHash({ ext: "image/jpeg", data }, 16, true, (error, data) =>
      error ? rej(error) : res(fromHexString(data))
    )
  );

export default async () => {
  return {
    infer: async (data) => ({ output: await phash(data) }),
  };
};
