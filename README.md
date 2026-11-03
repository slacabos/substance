# Substance

**A little world you can poke, stir, dig and brush.**

Seven materials. Three shapes. Turn a water orb into an ocean stretching to the horizon, comb a cube of fur, or dig a channel and watch the sea find its way in.

**[Play the live demo →](https://slacabos.github.io/substance/)** · [Run locally](#run-locally) · [Under the surface](#under-the-surface)

Built for touch and mouse with Three.js and WebGPU. You'll need a WebGPU-capable browser; unsupported browsers show a message instead.

## Start here

- **[Make some waves](https://slacabos.github.io/substance/?material=ocean&shape=plane&spin=off)** — hold, release, then drag to churn the sea. Tap the wind dial to build a storm.
- **[Meet the shore](https://slacabos.github.io/substance/?material=coast&shape=plane&spin=off)** — dig into the beach and connect a channel to the water.
- **[Comb a furry cube](https://slacabos.github.io/substance/?material=fur&shape=cube&setting=studio&spin=off)** — brush with your cursor or finger, then flick the coat.

## Pick your material

| Material  | What to try                                                                         |
| --------- | ----------------------------------------------------------------------------------- |
| **Water** | Tap for ripples, drag to stir, hold to drip.                                        |
| **Ocean** | Hold for bigger waves. Drag to stir up currents, foam and spray.                    |
| **Coast** | Make waves in the sea; dig and plough the beach. Open a channel and water flows in. |
| **Slime** | Press a dent, let go and watch it jiggle. Give it a spin for extra wobble.          |
| **Fur**   | Stroke to comb, press to part, flick to watch the hairs swing and settle.           |
| **Sand**  | Dig craters, plough trenches and flick grains. Steep slopes slide back down.        |
| **Fluff** | Brush a long purple coat, or spin the orb to make the strands bounce.               |

Every material works as an **Orb**, **Cube** or **Plane**. Switch shapes to watch the object unfurl into ground, with the camera following, then fold back again. The plane has a fixed view and a hazy horizon.

## Make it your own

Use the bottom panel to choose a material, shape and lighting: **Studio**, **Dusk** or **Night**. The play/pause button toggles idle spin; drag the background to spin an orb or cube yourself.

Ocean's wind dial cycles through **Beaufort 0–6**. Wind veers and gusts around the level you choose.

<details>
<summary><strong>Keyboard shortcuts</strong></summary>

| Key                   | Action                      |
| --------------------- | --------------------------- |
| `1` / `2` / `3` / `4` | Water / Ocean / Slime / Fur |
| `5` / `6` / `7`       | Coast / Sand / Fluff        |
| `O` / `C` / `P`       | Orb / Cube / Plane          |
| `B`                   | Cycle lighting              |
| `S`                   | Toggle idle spin            |

</details>

<details>
<summary><strong>Share a scene & tune the ocean</strong></summary>

The URL follows your scene. Copy it to share a view, or set it directly:

```text
?material=fur&shape=cube&setting=studio&spin=off
```

All parameters are optional. The defaults are Water, Orb, Night and spin on.

For finer ocean control, add these query parameters:

| Parameter         | Controls                     | Default |
| ----------------- | ---------------------------- | ------- |
| `oceanWind`       | Wind speed multiplier        | `0.90`  |
| `oceanWavelength` | Peak wavelength in orb radii | `0.23`  |
| `oceanAmplitude`  | Wave height multiplier       | `0.95`  |
| `oceanChop`       | Choppiness multiplier        | `2.00`  |
| `oceanFoam`       | Foam multiplier              | `1.40`  |

</details>

## Run locally

Requires **Node.js 20+**.

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. To check or build the project:

| Command                | Purpose                              |
| ---------------------- | ------------------------------------ |
| `npm test`             | Run unit tests with Vitest           |
| `npm run typecheck`    | Check TypeScript                     |
| `npm run lint`         | Run ESLint                           |
| `npm run format:check` | Check formatting                     |
| `npm run format`       | Apply Prettier formatting            |
| `npm run build`        | Typecheck and build into `dist/`     |
| `npm run preview`      | Preview the production build locally |

The [live site](https://slacabos.github.io/substance/) is hosted on GitHub Pages. Pushes to `main` deploy through [the Pages workflow](.github/workflows/pages.yml).

## Under the surface

Rendering uses Three.js `WebGPURenderer`. Custom shaders are written as **TSL node graphs**, and compute shaders drive waves, sand and hair.

| Code                                   | Responsibility                                            |
| -------------------------------------- | --------------------------------------------------------- |
| [`src/app.ts`](src/app.ts)             | Scene setup, material switching and the frame loop        |
| [`src/core/`](src/core/)               | Simulation math, tuning and CPU reference implementations |
| [`src/materials/`](src/materials/)     | Material implementations and GPU simulations              |
| [`src/nodes/`](src/nodes/)             | TSL shaders for surfaces, detail and shape transforms     |
| [`src/scene/`](src/scene/)             | Camera, renderer, lighting and fluid rendering            |
| [`src/interaction/`](src/interaction/) | Touch, pointer input and spin inertia                     |
| [`src/ui/`](src/ui/)                   | Pickers, controls and readouts                            |

<details>
<summary><strong>Waves, foam & splashes</strong></summary>

Ocean combines two kinds of motion: ambient **FFT waves** and an interactive wave field that responds to touch. A JONSWAP spectrum feeds GPU FFT cascades; changing wind gradually reshapes the sea, with short ripples responding before long swell.

The interaction simulation blends dispersive deep-water waves into nonlinear shallow-water flow near the coast. Currents carry foam and turbulence. GPU particles handle foam flecks, spray and droplets; dense splashes are reconstructed into a water surface with screen-space fluid rendering.

Coast adds a seabed, sandbar and incoming swell. Waves shoal, break and run up the beach. Land and sea share wetting and drying state, so a dug hole floods when a channel reaches the water.

Start with [`oceanFFT.ts`](src/materials/oceanFFT.ts), [`oceanSim.ts`](src/materials/oceanSim.ts), [`oceanFluid.ts`](src/materials/oceanFluid.ts) and [`fluidPass.ts`](src/scene/fluidPass.ts).

</details>

<details>
<summary><strong>Sand, fur & fluff</strong></summary>

Sand is a GPU height field. Neighbouring slopes settle toward the angle of repose, while digging and ploughing move sand into surrounding banks. Thrown grains take volume from the bed and return it when they land. Coast reuses the same sand engine and shading.

Fur simulates a small set of guides, then blends their motion across a much denser coat of camera-facing ribbons. Damped Verlet integration, bending resistance and body collision give the hairs weight and spring. Fluff uses longer, softer strands with its own simulation and purple palette.

Start with [`sandField.ts`](src/materials/sandField.ts), [`furSim.ts`](src/materials/furSim.ts) and [`fluff.ts`](src/materials/fluff.ts).

</details>

<details>
<summary><strong>One object, three shapes</strong></summary>

The cube is a superellipsoid driven by a spring. The plane maps the orb onto ground through a stereographic, log-polar chart, keeping the existing interaction simulations in orb space.

During the unfurl, nearby ground opens first, distant ground follows and the camera moves with it. Ambient waves and sand detail use ground coordinates on the plane, so their size stays consistent across the terrain. Setting-specific haze hides the chart's far edge.

Start with [`shape.ts`](src/nodes/shape.ts), [`chart.ts`](src/core/chart.ts) and [`transition.ts`](src/core/transition.ts).

</details>

Want to add a material? Implement [`Substance`](src/materials/types.ts), register it in [`App`](src/app.ts), and add a picker button in [`index.html`](index.html).

## License

[MIT](LICENSE) © 2026 Stefano Giudici.
