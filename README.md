# Substance

A mobile-first Three.js toy: a touchable 3D object you can switch between materials and shapes, then poke with a finger or mouse to see how each one reacts. It renders with WebGPU, so it needs a browser that supports it (recent Chrome, Edge or Safari); others get a short message instead.

## Materials

- **Water**: a clear, refractive blob. Tap for ripples, drag to stir, hold to drip.
- **Ocean**: a deep-blue miniature sea of FFT-simulated waves, with sun glitter, light glowing through crests and whitecaps where waves fold over. Holding a press loads the spherical free surface; releasing it sends a larger wave the longer it was held. Dragging stirs a tangential current and can build rough weather, whitewater, drifting foam, droplets and mist (the Beaufort readout follows the strongest patch).
- **Coast**: a separate miniature sea with raised islands, rocky banks and ridged high ground. Shoaling crests travel toward the irregular shores, collapse into whitewater and emit spray. The open Ocean stays free of land.
- **Slime**: press to dent it (hold to push deeper), let go and it jiggles. Spinning it hard wobbles it too.
- **Fur**: stroke to comb it (it stays combed, then slowly relaxes), press to part it, flick it to watch the hair swing and settle. Mouse hover brushes it.

Switch between **Orb** and **Cube** in the bottom panel; the shape morphs and every material works on both. The play/pause button starts or stops the slow idle spin; the three dots pick the setting: a white **Studio**, **Dusk** or **Night** (the default). The UI goes dark on the two dark settings. Drag the background to spin. On desktop, keys 1–5 switch material (5 selects Coast), O / C switch shape, B cycles the setting and S starts or stops the idle spin.

The URL follows the scene, so a reload (or a shared link) opens on the same view, e.g. `?material=fur&shape=cube&setting=studio&spin=off`. Each part is optional and defaults to water, orb, night and spin on.

Ocean uses wind speed **0.90×** (stronger wind builds higher seas), peak wavelength **0.23 orb radii**, wave height **0.95×**, choppiness **2.00×** and foam **1.40×** by default. Its surface has roughly **8k vertices on touch devices** and **24k on desktop**. The five wave values can still be overridden in a link with `oceanWind`, `oceanWavelength`, `oceanAmplitude`, `oceanChop` and `oceanFoam`; mesh detail is chosen by device.

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
- **Water, Ocean, Slime** use `MeshPhysicalNodeMaterial` surfaces displaced by TSL nodes (`nodes/deform.ts`). Water and Slime use interaction ring buffers; the ocean solves a wave field over the indexed sphere mesh and uploads its height, foam, turbulence and normal gradients each frame.
- **Ocean** waves are a Tessendorf FFT ocean on the GPU (`materials/oceanFFT.ts`): a JONSWAP spectrum with Donelan–Banner spreading in two or three cascades, inverse-FFT'd by Stockham compute passes into tiling, mipmapped displacement and slope textures. They're mapped onto the orb through equal-angle cube faces (`nodes/oceanWaves.ts`); every cascade tiles a face a whole number of times, so the equator joins seamlessly and only the polar faces blend in. Normals come per pixel, the slope variance the mips filter away becomes roughness (so tiny waves spread the sun into a glitter path), and whitecaps grow where the surface folds and linger for a few seconds. On top, the waves you make run on the GPU (`materials/oceanSim.ts`, with CPU references in `core/waveKernel.ts` and `core/shallowWater.ts`): iWave with a finite-depth operator fitted to the mesh, so ripples disperse while a tap's long ring runs out like a small tsunami, blended by depth into the nonlinear shallow-water equations near coasts, where waves steepen, surge and break. Taps push a volume-neutral footprint (down in the middle, up in a rim), taps that keep a steady beat push harder (`core/oceanRhythm.ts`), and the sea's volume is kept exact each frame. A tangential current carries foam and turbulence; whitecaps form on steep crests and splashes leave foam. Lagrangian foam flecks ride the current in a compute pass, and whitewater sites, read back from the GPU a few times a second, emit droplets and mist that return under radial gravity. An ocean-only sky reflection and water absorption shade the volume. This is a surface-flow model with particles, not a volumetric FLIP solver.
- **Coast** runs an independent ocean simulation. Its terrain removes inland water triangles and marks dry vertices as reflecting wave boundaries. Advancing shoaling crests displace the wet surface and generate surf foam at the bank; spray collides with the elevated terrain. Shore normals and distances share one attribute so the water stays within WebGPU's eight-buffer baseline. The shore mesh follows the orb-to-cube morph.
- **Fur** is about 36k simulated strands. A compute shader (`materials/furSim.ts`) moves each strand's chain of points every frame: Verlet integration with gravity and inertia, a stiffness pull toward a rest direction that combing and pressing lean over, collision with the body, and follow-the-leader length constraints so it stays stable however hard it's flicked.
- The strands are drawn in one instanced draw: camera-facing ribbons that curve through the simulated points, over a dark skin. They're never thinner than a pixel (thinner ones are drawn fainter) and are antialiased with alpha-to-coverage. Strands widen where the cube stretches the surface, so the coat stays even.
- **Cube** is a superellipsoid. Every shader maps a unit direction through `baseR()` in `nodes/shape.ts`, and a spring drives the morph uniform.

To add a material, implement `Substance` (`src/materials/types.ts`), register it in `App`, and add a button to the picker in `index.html`.
