# Substance

A mobile-first Three.js toy: a touchable 3D object you can switch between materials and shapes, then poke with a finger or mouse to see how each one reacts. It renders with WebGPU, so it needs a browser that supports it (recent Chrome, Edge or Safari); others get a short message instead.

## Materials

- **Water**: a clear, refractive blob. Tap for ripples, drag to stir, hold to drip.
- **Ocean**: rolling swells with foam on the crests. Tap to drop a stone; churn it to whip up a storm (the Beaufort readout climbs, then calms back down).
- **Slime**: press to dent it (hold to push deeper), let go and it jiggles. Spinning it hard wobbles it too.
- **Fur**: stroke to comb it (it stays combed, then slowly relaxes), press to part it, flick it to watch the hair swing and settle. Mouse hover brushes it.

Switch between **Orb** and **Cube** in the bottom panel; the shape morphs and every material works on both. The play/pause button starts or stops the slow idle spin; the three dots pick the setting: a white **Studio**, **Dusk** or **Night** (the default). The UI goes dark on the two dark settings. Drag the background to spin. On desktop, keys 1–4 switch material, O / C switch shape, B cycles the setting and S starts or stops the idle spin.

## Development

Requires Node 20+.

```sh
npm install
npm run dev        # local dev server with hot reload
npm run build      # typecheck + production build into dist/
npm test           # unit tests (Vitest)
npm run lint       # ESLint
npm run format     # Prettier
```

Pushes to `main` deploy to GitHub Pages through `.github/workflows/pages.yml` once Pages is set to "GitHub Actions" in the repository settings.

## How it's built

```
src/
  app.ts                 wires stage, materials, input and UI; runs the frame loop
  core/                  pure logic with unit tests: springs, ray/shape hit-testing
  scene/                 renderer + camera fit, backdrop settings, orb↔cube morph
  interaction/           pointer handling (touch vs spin) and spin inertia
  materials/             one class per material behind the `Substance` interface
  nodes/                 shaders, written in TSL (three.js node language)
  ui/                    material picker, shape and setting toggles, readout, keyboard
```

- Rendering is three.js `WebGPURenderer`. Every custom effect is a TSL node graph plugged into a node material, so there's no hand-written GLSL or WGSL.
- **Water, Ocean, Slime** share one `MeshPhysicalNodeMaterial` whose `positionNode` displaces the surface (`nodes/deform.ts`). Interaction writes ripple sources or spring-driven dents into uniform ring buffers; normals are rebuilt by finite differences so any displacement lights correctly.
- **Ocean** adds sharp-crested layered swells and a fragment pass for depth colour, subsurface tint and foam.
- **Fur** is about 36k simulated strands. A compute shader (`materials/furSim.ts`) moves each strand's chain of points every frame: Verlet integration with gravity and inertia, a stiffness pull toward a rest direction that combing and pressing lean over, collision with the body, and follow-the-leader length constraints so it stays stable however hard it's flicked.
- The strands are drawn in one instanced draw: camera-facing ribbons that curve through the simulated points, over a dark skin. They're never thinner than a pixel (thinner ones are drawn fainter) and are antialiased with alpha-to-coverage. Strands widen where the cube stretches the surface, so the coat stays even.
- **Cube** is a superellipsoid. Every shader maps a unit direction through `baseR()` in `nodes/shape.ts`, and a spring drives the morph uniform.

To add a material, implement `Substance` (`src/materials/types.ts`), register it in `App`, and add a button to the picker in `index.html`.
