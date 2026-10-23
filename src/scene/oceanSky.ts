import {
  DataTexture,
  EquirectangularReflectionMapping,
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
  type Texture,
} from 'three';
import type { PMREMGenerator } from 'three/webgpu';

/** A broad sky and soft sun for water reflections, separate from the studio lighting of other materials. */
export function oceanSkyReflection(pmrem: PMREMGenerator): Texture {
  const width = 256;
  const height = 128;
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const v = y / (height - 1);
    const horizon = Math.exp(-Math.pow((v - 0.48) / 0.16, 2));
    for (let x = 0; x < width; x++) {
      const u = x / (width - 1);
      const sun = Math.exp(-Math.pow((u - 0.72) / 0.085, 2) - Math.pow((v - 0.4) / 0.075, 2));
      const lower = Math.max(0, Math.min(1, (v - 0.5) * 4));
      const cloud = (Math.sin(u * 18 + Math.sin(u * 7) * 1.3) * Math.sin(v * 24 + u * 5) + 1) * 0.5;
      const diffuse = horizon * (0.8 + cloud * 0.2);
      const i = (y * width + x) * 4;
      pixels[i] = Math.min(255, 145 + diffuse * 75 + sun * 45 - lower * 68);
      pixels[i + 1] = Math.min(255, 178 + diffuse * 61 + sun * 34 - lower * 61);
      pixels[i + 2] = Math.min(255, 212 + diffuse * 41 + sun * 20 - lower * 49);
      pixels[i + 3] = 255;
    }
  }
  const source = new DataTexture(pixels, width, height, RGBAFormat, UnsignedByteType);
  source.colorSpace = SRGBColorSpace;
  source.mapping = EquirectangularReflectionMapping;
  source.needsUpdate = true;
  const target = pmrem.fromEquirectangular(source);
  source.dispose();
  return target.texture;
}
