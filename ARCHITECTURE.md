# Skjutbana – Current Architecture (dev branch snapshot)

> This document describes how the project is structured **right now in the `dev` branch**.
> It is meant to help a new chat/session or a new contributor understand:
>
> - what the runtime currently does
> - which modules exist and how they interact
> - what the current coordinate systems are
> - how hit events flow through the engine
> - where the current limitations and integration gaps are

---

# 1. High-level purpose

Skjutbana is a modular Python/Pygame application for a projected shooting range.

At a high level, the app does four things:

1. **Displays content** inside a configurable viewport
   - image
   - video
   - game scene
   - debug scenes

2. **Collects input / sensor data**
   - global mouse clicks
   - live camera frames
   - audio peak detection through microphone input

3. **Generates hit events**
   - directly from mouse clicks
   - indirectly from camera-based shot analysis

4. **Renders overlays**
   - hit markers
   - scanner debug/status overlays

The current entrypoint is tiny: `main.py` simply starts `App().run()`. The application shell and main loop live in `src/engine/app.py`. `App` starts the camera service and audio service immediately, enters `LoadingScene`, then on every frame updates the camera, updates the audio detector, forwards input to the current scene, updates the current scene, updates the hit scanner, and finally renders the scene plus overlays.  

---

# 2. Top-level structure

## Top-level files

- `main.py`
  - application entrypoint

- `config.py`
  - screen size
  - FPS
  - settings path
  - default viewport
  - some asset paths

- `ARCHITECTURE.md`
  - this document

- `README.md`
  - broad project overview

## Top-level directories

- `assets/`
  - images / movie assets

- `content/`
  - user/project content such as `menu.json`
  - settings persistence lives in `content/settings.json`

- `src/`
  - engine code

---

# 3. Runtime loop

## Main application loop

The current runtime lives in `src/engine/app.py`.

### Startup sequence

When `App` is created it currently:

1. initializes Pygame
2. creates the window
3. starts `camera_manager`
4. starts `audio_peak_detector`
5. creates and enters `LoadingScene`
6. syncs runtime services for the current scene
7. updates the window caption

### Per-frame loop

Each frame currently does this, in order:

1. `camera_manager.update()`
2. `audio_peak_detector.update()`
3. process Pygame events
4. pass each event to the current scene
5. update current scene
6. `hit_scanner.update(dt)`
7. update window caption
8. render current scene
9. `pygame.display.flip()`

### Scene switching

Scene switching is mediated by `SceneSwitch`, returned from scene handlers and scene updates.
When a switch happens, `App`:

1. calls `old_scene.on_exit()`
2. replaces `self.scene`
3. calls `new_scene.on_enter()`
4. re-syncs hit scanning based on scene flags
5. updates the caption

This is important because hit scanning is not simply “always on”. It is scene-driven.

---

# 4. Scene architecture

## Base scene contract

The base scene type lives in `src/engine/scene.py`.

A scene currently exposes:

- `on_enter()`
- `on_exit()`
- `handle_event(event)`
- `update(dt)`
- `render(screen)`

Two scene-level capability flags currently exist:

- `wants_hit_scanning`
- `wants_camera_preview`

At the moment, `App` uses `wants_hit_scanning` to enable or disable `hit_scanner`.  

## Scene factory

Menu items are converted into scenes by `src/engine/scene_factory.py`.

Current menu item → scene mappings:

- `video` → `VideoScene`
- `image` → `ImageScene`
- `game` → `GameScene`
- `transform_debug` → `TransformDebugScene`
- `settings` → `CalibrateViewportScene`
- `camera_scanport` → `CameraTestScene`
- `visual_hits_settings` → `VisualHitsSettingsScene`
- `scanner_debug_settings` → `ScannerDebugSettingsScene`

Most content scenes are wrapped in `OverlayScene`, which means they automatically gain the standard overlay pipeline.  

## Current scene types

### `LoadingScene`
Shows the loading screen image and transitions to `MenuScene` on keypress or mouse click.  

### `MenuScene`
Loads `content/menu.json` through `content_loader`, renders a folder/item-based menu, and launches scenes via `build_scene_from_item()`.  

### `ImageScene`
Displays a still image inside the current viewport.
Supports fit mode, panning, zoom, and reset.
`wants_hit_scanning = True`.  

### `VideoScene`
Displays a video inside the current viewport using `VideoPlayer`.
Supports fit mode, panning, zoom, pause, and reset.
`wants_hit_scanning = True`.  

### `GameScene`
Loads a game module dynamically through `game_loader`.
The game module is expected to define `create_game(game_root, viewport)`.
`wants_hit_scanning = True`.  

### `TransformDebugScene`
A grid/transform inspection scene for debugging hit coordinates and calibration assumptions.  

### `CalibrateViewportScene`
Manual viewport editor.
This is the current screen-space rectangle used as the drawing and hit-mapping area for content.  

### `CameraTestScene`
Live camera scene for adjusting the scanport rectangle.  

### `VisualHitsSettingsScene`
Edits visual hit settings:
- enabled/disabled
- fade/persistent mode
- lifetime

### `ScannerDebugSettingsScene`
Toggles the scanner debug overlay.  

---

# 5. Overlay architecture

## `OverlayScene`

This wrapper scene lives in `src/engine/scenes/overlay_scene.py`.

It is a crucial part of the current design:

- delegates real content rendering to the inner scene
- adds overlay rendering after the scene
- forwards `wants_hit_scanning` and `wants_camera_preview`
- clears the hit visualizer on exit
- injects a global mouse hit on left mouse click through `hit_input.push_mouse_hit(...)`

### Current render pipeline

Current order inside `OverlayScene.render(screen)`:

1. `inner.render(screen)`
2. `hit_visualizer.render(screen)`
3. `scanner_debug_overlay.render(screen)`
4. `scanner_status_overlay.render(screen)`

So scene content is always rendered first, and global overlays are layered on top.  

## Current overlays

### `HitVisualizer`
Global hit marker overlay.
Subscribes to `hit_input`.
Draws:
- mouse hits in red
- camera hits in cyan
Supports:
- fade mode
- persistent mode
- configurable lifetime
- configurable radius  

### `ScannerDebugOverlay`
Compact scanner debug panel.
Renders a small right-side diagnostic UI including warped/mask views and candidate/stable/known indicators, gated by `load_scanner_debug_overlay_enabled()`.  

### `ScannerStatusOverlay`
Larger status/coordinate debug overlay, also gated by `load_scanner_debug_overlay_enabled()`.
Shows:
- scanner state
- counts
- viewport/scanport/content rectangles
- last camera hit
- transformed coordinates
- current ring placement info  

---

# 6. Settings architecture

## Persistence model

Settings are stored in `content/settings.json` through helper functions in `src/engine/settings.py`.  

The settings module is currently responsible for both:
- loading/saving rectangles
- loading/saving feature settings

This means `settings.py` is a central configuration/service module, not just a schema file.

## Current persisted data categories

### Rectangles

#### `viewport`
The current projected drawing area on the app window.
Loaded with `load_viewport_rect()`.  

#### `scanport`
The region in full camera space that should be analyzed.
Loaded with `load_scanport_rect()`.  

#### `content_rect`
The rectangle where the actual content (image/video/tavla/game content) is considered to live.
Loaded with `load_content_rect()`.
**Important current behavior:** if `content_rect` is not stored, it falls back to the viewport.  

### Calibration

#### `camera_calibration`
A dict stored in settings, currently including homography if calibration has been run.
Loaded with `load_camera_calibration()`.  

### Visual hits settings

Stored under `visual_hits`, including:
- enabled
- mode (`fade` or `persistent`)
- lifetime
- radius  

### Scanner debug overlay settings

Stored under `scanner_debug_overlay`, currently mainly:
- enabled  

### Audio peak settings

Stored under `audio_peak`, including:
- threshold
- show_status_overlay

---

# 7. Coordinate systems

This is one of the most important current parts of the architecture.

The code now carries multiple coordinate spaces at the same time.

## 7.1 Full camera space

This is the coordinate system of the full raw camera frame.

Example source:
- `CameraManager.get_latest_frame()`

All camera-based hit detection ultimately starts here.  

## 7.2 Scanport-local space

The scanport is a rectangle inside the full camera frame.
Detection is performed on the crop defined by `scanport`.

So if:
- full camera point = `(camera_x, camera_y)`
- scanport rect = `(scanport.x, scanport.y, scanport.w, scanport.h)`

then scanport-local point is:

- `local_x = camera_x - scanport.x`
- `local_y = camera_y - scanport.y`

This distinction is critical in the current dev branch.  

## 7.3 Screen/app space

This is the app window coordinate system used by Pygame rendering.

The current hit pipeline stores:
- `screen_x`
- `screen_y`

These are the coordinates used for global overlays and visual hit markers.  

## 7.4 Viewport-local space

`HitInput` also derives viewport-local coordinates:

- `viewport_x = screen_x - viewport.x`
- `viewport_y = screen_y - viewport.y`

This is the local coordinate inside the viewport rectangle.  

## 7.5 Content-local space

`HitInput` also derives content-local coordinates:

- `content_x = screen_x - content_rect.x`
- `content_y = screen_y - content_rect.y`

and normalized content coordinates:

- `content_norm_x = content_x / content_rect.w`
- `content_norm_y = content_y / content_rect.h`

These are currently the closest thing to:
- image-space
- video-space
- tavla-space
- future game-space normalized coordinates

**Important current limitation:** the current `ImageScene` and `VideoScene` implement panning and zoom internally during render, but they do not obviously persist an updated `content_rect` back to settings when the user pans/zooms. So while `HitInput` supports `content_x/content_y/content_norm_x/content_norm_y`, those values only truly match what is on screen if `content_rect` is kept in sync by the scene or by future content-layout code.  

## 7.6 Homography space

There is still support for homography in `HitInput`.
Current logic prefers:

1. full camera → scanport normalized → viewport
2. fallback to homography
3. fallback to raw camera values

So the current architecture has shifted toward scanport normalization as the primary mapping path, with homography still available as a fallback if present.  

---

# 8. Hit architecture

## `HitEvent`

The canonical hit event currently lives in `src/engine/input/hit_input.py`.

Current fields:

- `source`
- `screen_x`
- `screen_y`
- `viewport_x`
- `viewport_y`
- `content_x`
- `content_y`
- `content_norm_x`
- `content_norm_y`
- `camera_x`
- `camera_y`
- `timestamp`

This is the project’s key cross-layer data structure.

It is important because it lets the same event be used by:
- overlays
- debug tools
- gameplay
- future scoring logic
- future particle logic
- future weapon logic

## `HitInput`

`HitInput` is the current central hit dispatcher.

It currently provides:

- `push_mouse_hit(screen_x, screen_y)`
- `push_camera_hit(camera_x, camera_y)`
- `subscribe(callback)`
- `unsubscribe(callback)`
- `poll()`
- `reload_calibration()`

### Mouse path

A mouse click becomes a hit event directly in app/screen space.
If inverse homography exists, it also estimates a camera-space point.

### Camera path

A camera hit is currently mapped to screen/app coordinates through:

1. scanport normalization → viewport
2. fallback homography
3. raw fallback

Then `viewport_x`, `content_x`, and normalized content coordinates are derived from the final screen point.  

### Global state

`HitInput` also stores:

- `last_mouse_hit`
- `last_camera_hit`
- `last_hit`

This is what current debug overlays use to show the latest transformed hit.  

---

# 9. Camera architecture

## `CameraManager`

The camera service lives in `src/engine/camera/camera_manager.py`.

It is currently responsible for:

- opening the camera
- applying preferred width/height/fps
- probing camera capabilities
- grabbing the latest frame
- storing a timestamped `CameraFrame`

Current preferred capture mode is:
- 3840x2160
- 30 fps

when supported by the device.  

## Related camera helper layer

`CameraManager` depends on camera capability helpers:
- `probe_camera_capabilities(...)`
- `apply_preferred_camera_settings(...)`

These are part of the camera hardware abstraction layer.  

## Current camera state model

`CameraManager` maintains:
- `cap`
- `latest_frame`
- `capabilities`
- `property_apply_result`
- `last_error`
- `running`

The rest of the engine mostly treats `camera_manager` as a singleton service.  

---

# 10. Audio architecture

## `AudioPeakDetector`

The current audio detector is a lightweight peak detector, not a full classifier.

It lives in `src/engine/audio/audio_peak_detector.py`.

### Purpose

Its role is currently:
- detect loud transient peaks from the mic
- produce `AudioPeakEvent`
- act as a timing signal for visual analysis

It is **not yet** a weapon classifier or speech recognizer.  

### Backend strategy

It currently tries:
1. PulseAudio default device
2. ALSA default

using `ffmpeg` as the audio capture backend.  

### Current peak logic

It currently uses:
- absolute peak threshold
- adaptive noise floor
- peak ratio against noise floor
- cooldown

### Current public API

- `start()`
- `stop()`
- `update()` (currently no-op)
- `get_events_since(ts)`
- `get_latest_event()`
- `get_status_lines()`
- `get_waveform_snapshot()`
- `get_peak_threshold()`
- `set_peak_threshold()`

So in the current architecture, the audio subsystem is running continuously in the background and exposing peak events as a shared service.  

---

# 11. Scanner architecture

## `HitScanner`

`App` imports `hit_scanner` and updates it every frame after the current scene update.  

Architecturally, the scanner is intended to be:

- enabled only when the current scene wants hit scanning
- fed by:
  - latest camera frames
  - audio peak timing
- responsible for:
  - selecting a single best camera hit candidate
  - forwarding that candidate through `hit_input.push_camera_hit(...)`

## Current integration state

From the files around it, the scanner is clearly expected to produce:
- candidate lists
- stable tracks
- known holes
- debug frames
- status snapshot
- last/best candidate data

because both `scanner_debug_overlay` and `scanner_status_overlay` expect those structures.  

That means the scanner is architecturally not just a detector — it is also currently the source of most scanner-side debug state.  

## Important current note

The scanner area is still the most experimental part of the current dev architecture.
The surrounding architecture assumes that the scanner can expose a rich debug snapshot.
This is useful, but it also means:
- scanner implementation changes can easily break overlays
- scanner debug fields form an implicit contract with overlay code

That contract should ideally be formalized later.  

---

# 12. Video and image rendering architecture

## Shared pattern

Both `ImageScene` and `VideoScene` currently:

- load the viewport on enter
- render content clipped to the viewport
- support fit modes (`stretch`, `contain`, `cover`)
- support panning via offsets
- support zoom
- reset with `R`
- escape back to menu
- set `wants_hit_scanning = True`

## `VideoPlayer`

Video playback is done through OpenCV + Pygame surface conversion in `src/engine/video_player.py`.

Current responsibilities:
- open movie
- step frames according to source FPS
- convert BGR → RGB
- create Pygame surfaces
- handle pause/finished state

This is a self-contained media helper, not a general scene class.  

## Current architectural implication

Image/video rendering and hit detection are not yet fully coupled through a shared “content layout service”.
Scenes render with internal offsets/zoom, but the hit system relies on `content_rect` from settings.
So there is still a gap between:
- what is actually drawn
- what hit mapping assumes is drawn

That is one of the key architecture gaps in the current dev state.  

---

# 13. Game architecture

## `GameScene`

`GameScene` is a host scene for pluggable game modules.

It:
- loads viewport
- dynamically imports a Python game script
- requires the module to expose `create_game(game_root, viewport)`
- delegates `handle_event`, `update`, and `render`

So the current game engine is a script-host model, not a built-in ECS or object hierarchy yet.  

## `game_loader`

Dynamic import is handled by `src/engine/game_loader.py`.

This means games can currently be built as separate scripts without changing the engine core.  

## Current implication

This is a strong extension point:
- gameplay logic can already be externalized
- hit events can already be fed into game logic
- future weapon/material/particle systems can likely plug into this layer rather than the camera layer

---

# 14. Menu/content architecture

## `content/menu.json`

The menu is data-driven.
`MenuScene` loads it through `content_loader`.

The menu defines:
- folders
- items
- preview images
- scene types
- paths/scripts for content

This means content selection is already externalized from Python code.  

## Architectural consequence

This is a good base for:
- more scene types later
- game content packs
- training packs
- scenario packs
- calibration tools accessible from menu data

---

# 15. Current architectural strengths

These are the strongest parts of the current dev architecture:

## 15.1 Shared event model
`HitEvent` is already rich enough to support:
- debug
- overlays
- scene logic
- future gameplay systems

## 15.2 Scene wrapper pattern
`OverlayScene` cleanly separates content rendering from overlay rendering.

## 15.3 Service singletons
The current services are centralized and easy to access:
- `camera_manager`
- `audio_peak_detector`
- `hit_input`
- `hit_scanner`
- `hit_visualizer`

## 15.4 Data-driven menu
The menu/content selection layer is already externalized.

## 15.5 Settings as persistent runtime state
Viewport, scanport, content rect, calibration, and visualization settings are already centralized in one persistence module.  

---

# 16. Current architectural weaknesses / gaps

This section is intentionally blunt and practical.

## 16.1 Content layout is not yet a first-class service
The engine now has `content_rect`, which is good, but scenes like `ImageScene` and `VideoScene` still manage offsets/zoom internally during render and do not obviously publish their final on-screen content rectangle back into the central settings/runtime state.  

This means:
- `content_x/content_y/content_norm_x/content_norm_y` exist
- but can drift from what is actually on screen

## 16.2 Scanner debug contract is implicit
Overlays assume a lot about what the scanner exposes.
There is no explicit schema object for the debug snapshot yet.

## 16.3 Mouse hit injection is global in `OverlayScene`
Any wrapped scene receives a global left-click → `push_mouse_hit(...)` before its own handler logic.
This is convenient, but it means mouse-hit behavior is partially global rather than purely scene-local.  

## 16.4 Audio output, weapon metadata, and content-aware targeting are not yet integrated
The architecture is ready to support them conceptually, but they are not part of the current runtime contracts yet.  

## 16.5 Projection/content synchronization is not yet sessionized
The engine has viewport, scanport, and calibration, but does not yet have a formal “session startup” pipeline for:
- sync image
- fresh calibration
- white baseline capture
- content-aware reference setup

That is a major future architecture addition.  

---

# 17. Recommended future architecture direction

This section is not “what exists now”; it is the cleanest next direction based on the current structure.

## 17.1 Promote content layout to a runtime service
The engine should likely gain a dedicated content layout state that scenes update when they pan/zoom/move content.

That would make:
- `content_rect`
- normalized content coordinates
- hit-to-content mapping

fully reliable.

## 17.2 Formalize scanner snapshot schema
Instead of free-form dicts, define a scanner debug snapshot model with a stable structure.

## 17.3 Add session startup pipeline
For every new piece of content:
1. project sync image
2. verify/calibrate geometry
3. project white frame
4. capture session reference
5. enter live content mode

## 17.4 Add candidate plotting as a first-class debug layer
This is one of the most important next steps because it will make the scanner observable.

## 17.5 Keep gameplay interpretation separate from physical detection
The current architecture already leans this way, and that is good.

The scanner should determine:
- where the physical impact likely was

The game layer should determine:
- what that means in gameplay
- particles
- material response
- weapon behavior
- damage / penetration

---

# 18. Practical file map

A new contributor/chat can use this as a starting point.

## Entrypoints / shell
- `main.py`
- `src/engine/app.py`

## Global contracts
- `src/engine/scene.py`
- `src/engine/input/hit_input.py`
- `src/engine/settings.py`

## Sensor services
- `src/engine/camera/camera_manager.py`
- `src/engine/audio/audio_peak_detector.py`
- `src/engine/camera/hit_scanner.py`

## Rendering / overlay
- `src/engine/scenes/overlay_scene.py`
- `src/engine/visual/hit_visualizer.py`
- `src/engine/visual/scanner_debug_overlay.py`
- `src/engine/visual/scanner_status_overlay.py`

## Scene implementations
- `src/engine/scenes/loading.py`
- `src/engine/scenes/menu.py`
- `src/engine/scenes/image.py`
- `src/engine/scenes/video.py`
- `src/engine/scenes/game.py`
- `src/engine/scenes/transform_debug.py`
- `src/engine/scenes/camera_test.py`
- `src/engine/scenes/calibrate.py`
- `src/engine/scenes/visual_hits_settings.py`
- `src/engine/scenes/scanner_debug_settings.py`

## Dynamic content / game loading
- `src/engine/content_loader.py`
- `src/engine/scene_factory.py`
- `src/engine/game_loader.py`
- `src/engine/video_player.py`

## Data/config
- `config.py`
- `content/menu.json`
- `content/settings.json`

---

# 19. Current architecture summary in one paragraph

The current `dev` branch is a Pygame application with a scene-based runtime, persistent viewport/scanport/content settings, a central `HitInput` event dispatcher, global hit overlays, a continuously running camera service and audio peak service, and an experimental camera-based hit scanner that is turned on only for scenes that request it. Content scenes (image/video/game) are wrapped in `OverlayScene`, which provides shared hit visualization and scanner debugging. The engine already carries multiple coordinate systems (camera, scanport, viewport, content) in `HitEvent`, but the mapping between rendered content and persisted `content_rect` is not yet fully first-class. In other words: the current architecture is already modular and extensible, but the next big structural win will come from making detection more observable and making content geometry/session setup more explicit.

