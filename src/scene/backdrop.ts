import {
  AdditiveBlending,
  BackSide,
  Color,
  Mesh,
  PlaneGeometry,
  ShaderMaterial,
  SphereGeometry,
} from 'three';
import backdropFrag from '../shaders/backdrop.frag.glsl?raw';
import backdropVert from '../shaders/backdrop.vert.glsl?raw';
import poolFrag from '../shaders/pool.frag.glsl?raw';
import poolVert from '../shaders/pool.vert.glsl?raw';

/** A big inside-out sphere painted as a dark studio. It's opaque, so transmissive materials refract it. */
export function createBackdrop(): Mesh {
  const mesh = new Mesh(
    new SphereGeometry(40, 64, 32),
    new ShaderMaterial({
      side: BackSide,
      depthWrite: false,
      vertexShader: backdropVert,
      fragmentShader: backdropFrag,
    }),
  );
  mesh.renderOrder = -1;
  return mesh;
}

export interface LightPool {
  mesh: Mesh;
  set(color: string, strength: number): void;
}

/** A soft additive glow on the floor under the object, tinted per material. */
export function createLightPool(): LightPool {
  const uniforms = { uColor: { value: new Color() }, uA: { value: 0.2 } };
  const mesh = new Mesh(
    new PlaneGeometry(3.2, 3.2),
    new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
      uniforms,
      vertexShader: poolVert,
      fragmentShader: poolFrag,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = -1.55;
  return {
    mesh,
    set(color, strength) {
      uniforms.uColor.value.set(color);
      uniforms.uA.value = strength;
    },
  };
}
