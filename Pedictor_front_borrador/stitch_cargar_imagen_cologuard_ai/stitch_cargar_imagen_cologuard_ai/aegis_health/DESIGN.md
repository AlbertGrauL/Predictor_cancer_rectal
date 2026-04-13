# Design System Document

## 1. Overview & Creative North Star: "The Clinical Sanctuary"
This design system rejects the cold, sterile aesthetic of traditional medical software in favor of a "Clinical Sanctuary." Our Creative North Star is defined by **Empathy through Clarity**. We move beyond generic templates by utilizing high-end editorial layouts, intentional white space, and a sophisticated layering of surfaces. 

Instead of rigid grids and harsh borders, we use **asymmetric balance** and **tonal depth** to guide the user’s eye. The interface should feel like a premium, quiet consultation room—authoritative yet deeply calming. We achieve this by prioritizing breathable layouts, oversized editorial headers, and a "soft-touch" digital finish.

---

## 2. Colors & Tonal Architecture
The palette is built on "Trustworthy Blues" and "Calming Greens," but its sophistication comes from how these tones are layered.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to section content. Boundaries must be defined strictly through background color shifts. For example, a `surface-container-low` card should sit on a `surface` background to create a boundary through contrast, not lines.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. Use the following tiers to define importance:
- **Base Layer:** `surface` (#f9f9fd) for the overall application backdrop.
- **Mid Layer:** `surface-container-low` (#f3f3f7) for secondary content areas or sidebars.
- **Top Layer:** `surface-container-lowest` (#ffffff) for primary interaction cards or modals.

### The Glass & Gradient Rule
To prevent a "flat" feel, use **Glassmorphism** for floating elements (e.g., Navigation bars, Tooltips).
- **Glass Specs:** Background color `surface` at 70% opacity with a `backdrop-blur` of 20px.
- **Signature Textures:** For high-impact CTAs or Hero sections, use a subtle linear gradient from `primary` (#00478d) to `primary_container` (#005eb8) at a 135-degree angle. This adds a "lithographic" depth that flat color lacks.

---

## 3. Typography: Editorial Authority
We pair **Manrope** (Display/Headlines) with **Inter** (Body/UI) to create a high-contrast, editorial hierarchy that feels both modern and medical.

| Level | Token | Font | Size | Weight | Intent |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display** | `display-lg` | Manrope | 3.5rem | 700 | Large data summaries / Hero moments. |
| **Headline**| `headline-md`| Manrope | 1.75rem | 600 | Section headers and diagnostic titles. |
| **Title**   | `title-lg`   | Inter   | 1.375rem| 600 | Card titles and modal headers. |
| **Body**    | `body-lg`    | Inter   | 1.0rem  | 400 | Patient reports and general instruction. |
| **Label**   | `label-md`   | Inter   | 0.75rem | 500 | Metadata and small button text. |

*Note: Use `on-surface-variant` (#424752) for body text to reduce eye strain while maintaining WCAG AA compliance.*

---

## 4. Elevation & Depth: Tonal Layering
We do not use structural lines. We use light.

- **The Layering Principle:** Depth is achieved by stacking. A `surface-container-lowest` card placed on a `surface-container` background provides a natural, soft lift.
- **Ambient Shadows:** For elevated components (e.g., a "Result Card"), use an extra-diffused shadow: `box-shadow: 0 12px 32px rgba(25, 28, 30, 0.06);`. The shadow color is a tinted version of `on-surface` to mimic natural light.
- **The "Ghost Border" Fallback:** If a container requires more definition for accessibility, use the `outline-variant` (#c2c6d4) at **15% opacity**. Never use 100% opaque borders.
- **Glassmorphism:** Use for persistent elements like the Navigation Bar. This allows content to bleed through, making the layout feel integrated and "airy."

---

## 5. Components

### Medical-Themed Buttons
- **Primary:** Gradient from `primary` to `primary_container`. `xl` roundedness (1.5rem). High-contrast `on_primary` text.
- **Secondary (Calming):** `secondary_container` background with `on_secondary_container` text. This is for non-urgent medical actions.
- **Tertiary (Neutral):** No background, `primary` text. Use for "Cancel" or "Go Back."

### Upload Dropzones (High Intent)
- **Style:** Instead of dashed lines, use a `surface-container-high` background with a `md` (0.75rem) rounded corner. 
- **State:** When a file is hovered, transition the background to `primary_fixed` (#d6e3ff) to provide immediate tactile feedback.

### Result Cards (The Data Core)
- **Structure:** No dividers. Use `Spacing 6` (2rem) between the header and the results. 
- **Context:** Use a vertical accent bar on the left side using `tertiary` (#0f5238) for "Normal" results or `error` (#ba1a1a) for "Requires Attention." This bar should be 4px wide with `full` rounding.

### Typography Scales & Lists
- **Rule:** Forbid divider lines in lists. Separate list items using `Spacing 3` (1rem) and subtle background shifts (alternating between `surface` and `surface-container-low`) to maintain a clean, high-end medical journal aesthetic.

### Additional Recommended Components
- **Confidence Gauges:** A custom progress bar using `tertiary_container` for the track and `tertiary` for the fill, indicating the AI's detection confidence.
- **Phase Indicators:** Large, soft-rounded steps for the detection process (Upload -> Processing -> Analysis -> Result).

---

## 6. Do’s and Don’ts

### Do
- **Do** use `Spacing 16` and `20` to create "Editorial Breathing Room" between major sections.
- **Do** use `xl` (1.5rem) rounded corners for primary containers to evoke a "friendly/approachable" feel.
- **Do** use `tertiary` (Green) for all health-positive messaging to reinforce a sense of calm and success.

### Don’t
- **Don’t** use pure black (#000000). Always use `on-surface` (#191c1e) for text.
- **Don’t** use default browser focus rings. Use a 2px `surface-tint` offset for accessibility.
- **Don’t** use hard shadows. If you can clearly see where the shadow ends, it is too dark.
- **Don’t** use "Alert Red" for everything. Reserve the `error` token only for critical diagnostic findings.