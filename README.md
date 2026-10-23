# Substance

A mobile-first Three.js toy: a touchable 3D object you can switch between materials and shapes, then poke with a finger or mouse to see how each one reacts.

## Materials

- **Water**: a clear, refractive blob. Tap for ripples, drag to stir, hold to drip.
- **Ocean**: rolling swells with foam on the crests. Tap to drop a stone; churn it to whip up a storm (the Beaufort readout climbs, then calms back down).
- **Slime**: press to dent it (hold to push deeper), let go and it jiggles. Spinning it hard wobbles it too.
- **Hairy**: stroke to comb the fur, press to part it, spin to fluff it. Mouse hover brushes it.

Switch between **Orb** and **Cube** at the top right; the shape morphs and every material works on both. Drag the background to spin. On desktop, keys 1–4 switch material and O / C switch shape.

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
  scene/                 renderer + camera fit, studio backdrop, orb↔cube morph
  interaction/           pointer handling (touch vs spin) and spin inertia
  materials/             one class per material behind the `Substance` interface
  shaders/               GLSL, imported as raw strings
  ui/                    material picker, shape toggle, readout, keyboard
```

- **Water, Ocean, Slime** share one `MeshPhysicalMaterial` with vertex displacement injected through `onBeforeCompile` (`shaders/deform.vert.glsl`). Interaction writes ripple sources or spring-driven dents into uniform ring buffers; normals are rebuilt by finite differences so any displacement lights correctly.
- **Ocean** adds sharp-crested layered swells and a fragment pass for depth colour, subsurface tint and foam.
- **Fur** is shell texturing: one instanced draw, one shell per instance, strands cut from a 3D lattice so density stays even on the cube.
- **Cube** is a superellipsoid. Every shader maps a unit direction through `baseR()` in `shaders/shape.glsl`, and a spring drives the morph uniform.

To add a material, implement `Substance` (`src/materials/types.ts`), register it in `App`, and add a button to the picker in `index.html`.
