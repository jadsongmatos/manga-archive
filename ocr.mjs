import { createOCREngine } from "tesseract-wasm";
import { loadWasmBinary } from "tesseract-wasm/node";
import fetch from "node-fetch";
import { readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import sharp from "sharp";

async function loadImage(path) {
  const image = await sharp(path).ensureAlpha();
  const { width, height } = await image.metadata();
  return {
    data: await image.raw().toBuffer(),
    width,
    height,
  };
}

async function loadModel() {
  const modelPath = "eng.traineddata";
  if (!existsSync(modelPath)) {
    console.log("Downloading text recognition model...");
    const modelURL =
      "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata";
    const response = await fetch(modelURL);
    if (!response.ok) {
      process.stderr.write(`Failed to download model from ${modelURL}`);
      process.exit(1);
    }
    const data = await response.arrayBuffer();
    await writeFile(modelPath, new Uint8Array(data));
  }
  return readFile("eng.traineddata");
}

const wasmBinary = await loadWasmBinary();
const engine = await createOCREngine({ wasmBinary });

console.log("load model...")
const modelLoaded = await loadModel().then((model) => engine.loadModel(model));
console.log(engine.getVariable("tessedit_pageseg_mode"))
engine.setVariable("tessedit_pageseg_mode","3")
engine.setVariable("thresholding_method","1")
//engine.setVariable("noise_cert_factor","0.9")
console.log(engine.getVariable("tessedit_pageseg_mode"))

console.log("load image...")
const image = await loadImage("./1.jpg");
await engine.loadImage(image);

const text = await engine.getText((progress) => {
  process.stderr.write(`\r${progress}%...\n`);
});

console.log(text)