const IMAGE_SIZE = 224;

const EMBEDDING_NODES = {
  "1.00": "module_apply_default/MobilenetV1/Logits/global_pool",
  "2.00": "module_apply_default/MobilenetV2/Logits/AvgPool",
};

const MODEL_INFO = {
  "1.00": {
    0.25: {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/1",
      inputRange: [0, 1],
    },
    "0.50": {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/1",
      inputRange: [0, 1],
    },
    0.75: {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/1",
      inputRange: [0, 1],
    },
    "1.00": {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1",
      inputRange: [0, 1],
    },
  },
  "2.00": {
    "0.50": {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/2",
      inputRange: [0, 1],
    },
    0.75: {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/2",
      inputRange: [0, 1],
    },
    "1.00": {
      url: "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2",
      inputRange: [0, 1],
    },
  },
};

// See ModelConfig documentation for expectations of provided fields.
export async function load(
  tf,
  modelConfig = {
    version: 1,
    alpha: 1.0,
  }
) {
  if (tf == null) {
    throw new Error(
      `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`
    );
  }
  const versionStr = modelConfig.version.toFixed(2);
  const alphaStr = modelConfig.alpha ? modelConfig.alpha.toFixed(2) : "";
  let inputMin = -1;
  let inputMax = 1;
  // User provides versionStr / alphaStr.
  if (modelConfig.modelUrl == null) {
    if (!(versionStr in MODEL_INFO)) {
      throw new Error(
        `Invalid version of MobileNet. Valid versions are: ` +
          `${Object.keys(MODEL_INFO)}`
      );
    }
    if (!(alphaStr in MODEL_INFO[versionStr])) {
      throw new Error(
        `MobileNet constructed with invalid alpha ${modelConfig.alpha}. Valid ` +
          `multipliers for this version are: ` +
          `${Object.keys(MODEL_INFO[versionStr])}.`
      );
    }
    [inputMin, inputMax] = MODEL_INFO[versionStr][alphaStr].inputRange;
  }
  // User provides modelUrl & optional<inputRange>.
  if (modelConfig.inputRange != null) {
    [inputMin, inputMax] = modelConfig.inputRange;
  }
  const mobilenet = new MobileNetImpl(
    tf,
    versionStr,
    alphaStr,
    modelConfig.modelUrl,
    inputMin,
    inputMax
  );
  await mobilenet.load();
  return mobilenet;
}

class MobileNetImpl {
  model;

  // Values read from images are in the range [0.0, 255.0], but they must
  // be normalized to [min, max] before passing to the mobilenet classifier.
  // Different implementations of mobilenet have different values of [min, max].
  // We store the appropriate normalization parameters using these two scalars
  // such that:
  // out = (in / 255.0) * (inputMax - inputMin) + inputMin;
  normalizationConstant;

  constructor(tf, version, alpha, modelUrl, inputMin = -1, inputMax = 1) {
    this.version = version;
    this.alpha = alpha;
    this.modelUrl = modelUrl;
    this.inputMin = inputMin;
    this.inputMax = inputMax;
    this.tf = tf;
    this.normalizationConstant = (inputMax - inputMin) / 255.0;
  }

  async load() {
    if (this.modelUrl) {
      this.model = await this.tf.loadGraphModel(this.modelUrl);
      // Expect that models loaded by URL should be normalized to [-1, 1]
    } else {
      const url = MODEL_INFO[this.version][this.alpha].url;
      this.model = await this.tf.loadGraphModel(url, { fromTFHub: true });
    }

    // Warmup the model.
    const result = this.tf.tidy(() =>
      this.model.predict(this.tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]))
    );
    await result.data();
    result.dispose();
  }

  /**
   * Computes the logits (or the embedding) for the provided image.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   *     video, or canvas.
   * @param embedding If true, it returns the embedding. Otherwise it returns
   *     the 1000-dim logits.
   */
  infer(img, embedding = false) {
    return this.tf.tidy(() => {
      if (!(img instanceof this.tf.Tensor)) {
        img = this.tf.browser.fromPixels(img);
      }

      // Normalize the image from [0, 255] to [inputMin, inputMax].
      const normalized = this.tf.add(
        this.tf.mul(this.tf.cast(img, "float32"), this.normalizationConstant),
        this.inputMin
      );

      // Resize the image to
      let resized = normalized;
      if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = this.tf.image.resizeBilinear(
          normalized,
          [IMAGE_SIZE, IMAGE_SIZE],
          alignCorners
        );
      }

      // Reshape so we can pass it to predict.
      const batched = this.tf.reshape(resized, [-1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      let result;

      if (embedding) {
        const embeddingName = EMBEDDING_NODES[this.version];
        const internal = this.model.execute(batched, embeddingName);
        result = this.tf.squeeze(internal, [1, 2]);
      } else {
        const logits1001 = this.model.predict(batched);
        // Remove the very first logit (background noise).
        result = this.tf.slice(logits1001, [0, 1], [-1, 1000]);
      }

      return result;
    });
  }
}
