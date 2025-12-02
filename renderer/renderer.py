# renderer.py
import math
import pygame
import numpy as np
import typing
from renderer import test

if typing.TYPE_CHECKING:
    from renderer.cameras import FirstPersonCamera

FPS = 60


class Renderer3D:
    def __init__(self, width=1000, height=700, title="First-Person Demo"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.fast_text_renderer = FastTextRenderer("Arial", 20, (255, 255, 255), True)

    def set_text_renderer_settings(
        self, font_path=None, size=20, color=(255, 255, 255), antialias=True
    ):
        self.fast_text_renderer = FastTextRenderer(font_path, size, color, antialias)

    def clear(self):
        self.screen.fill((80, 150, 255))  # sky

    # def draw_ground_grid(self, cam: "FirstPersonCamera"):
    #     # simple ground at y = 0
    #     color1 = (160, 160, 160)
    #     color2 = (130, 130, 130)

    #     for gx in range(-20, 21):
    #         for gz in range(-20, 21):
    #             corners_world = [
    #                 (gx, 0, gz),
    #                 (gx + 1, 0, gz),
    #                 (gx + 1, 0, gz + 1),
    #                 (gx, 0, gz + 1),
    #             ]
    #             corners_screen = []
    #             for c in corners_world:
    #                 cam_space = cam.world_to_camera(c)
    #                 sp = cam.camera_to_screen(cam_space)
    #                 if sp is None:
    #                     break
    #                 corners_screen.append(sp)
    #             if len(corners_screen) == 4:
    #                 color = color1 if (gx + gz) % 2 == 0 else color2
    #                 pygame.draw.polygon(self.screen, color, corners_screen)
    def draw_ground_grid(self, cam: "FirstPersonCamera"):
        # return self.draw_ground_quad(cam)
        """Optimized infinite ground grid at y=0."""

        # ---- SETTINGS ----
        GRID_RADIUS = 10  # how far around the player to draw
        CELL_SIZE = 1.0
        color1 = (160, 160, 160)
        color2 = (130, 130, 130)

        cx, _, cz = cam.position
        yaw = cam.yaw

        # ---- PRECOMPUTE DIRECTION VECTORS ----
        # forward direction on XZ plane
        fwd_x = math.sin(yaw)
        fwd_z = math.cos(yaw)

        # 90° left/right for frustum culling
        culling_angle = math.radians(65)
        left_x = math.sin(yaw - culling_angle)
        left_z = math.cos(yaw - culling_angle)
        right_x = math.sin(yaw + culling_angle)
        right_z = math.cos(yaw + culling_angle)

        for ix in range(-GRID_RADIUS, GRID_RADIUS + 1):
            for iz in range(-GRID_RADIUS, GRID_RADIUS + 1):
                # world position of the center of this cell
                wx = int(cx) + ix * CELL_SIZE
                wz = int(cz) + iz * CELL_SIZE

                # vector from camera → cell center
                dx = wx - cx
                dz = wz - cz

                # ---- FRUSTUM CULLING (2D) ----
                # dot with forward must be positive (in front of camera)
                dot_fwd = dx * fwd_x + dz * fwd_z
                if dot_fwd < -1:  # behind camera → skip
                    continue

                # dot with left/right must be within FOV
                # ~65° half-FOV → cos(65°) ≈ 0.42
                # larger threshold = wider visible range
                side_dot_left = dx * left_x + dz * left_z
                side_dot_right = dx * right_x + dz * right_z
                if side_dot_left < -15 or side_dot_right < -15:
                    continue

                # ---- PROJECT ONLY THIS CELL ----
                corners = [
                    (wx, 0, wz),
                    (wx + CELL_SIZE, 0, wz),
                    (wx + CELL_SIZE, 0, wz + CELL_SIZE),
                    (wx, 0, wz + CELL_SIZE),
                ]

                screen_pts = []
                for c in corners:
                    cam_p = cam.world_to_camera(c)
                    sp = cam.camera_to_screen(cam_p)
                    if sp is None:
                        break
                    screen_pts.append(sp)

                if len(screen_pts) == 4:
                    color = color1 if ((int(wx) + int(wz)) % 2 == 0) else color2
                    pygame.draw.polygon(self.screen, color, screen_pts)

    def draw_ground_grid_lines(self, cam, cell_size=1.0, radius=40, major_every=5):
        """
        Fast grid-line renderer.
        - cell_size: world units between minor grid lines (use 2 or 4 for larger, faster grid)
        - radius: how many cells from camera center to consider (world-space)
        - major_every: draw thicker major lines every N cells for distance cues
        """

        # colors
        minor_col = (140, 140, 140)
        major_col = (200, 200, 200)
        axis_col = (200, 120, 120)  # optional X-axis highlight
        w, h = self.width, self.height

        # camera stuff
        cx, cy, cz = cam.position  # camera world position
        yaw = cam.yaw
        # Precompute forward dot for fast behind-camera culling (XZ plane)
        fwd_x = math.sin(yaw)
        fwd_z = math.cos(yaw)

        # limit distances (in world units)
        max_draw_dist = radius * cell_size

        # compute range of integer grid indices to draw
        # center indices (integer grid coords)
        center_ix = int(math.floor(cx / cell_size))
        center_iz = int(math.floor(cz / cell_size))

        # We'll draw lines at world X = (ix * cell_size) for ix in [center_ix - radius .. +radius]
        ix_min = center_ix - radius
        ix_max = center_ix + radius
        iz_min = center_iz - radius
        iz_max = center_iz + radius

        # Precompute a world-plane 'far' bounds (we'll clamp to max_draw_dist)
        # We create each line as between two world points sufficiently far along the other axis.
        # For X-lines (vertical in world Z), vary Z from cz - max_draw_dist .. cz + max_draw_dist
        z0 = cz - max_draw_dist
        z1 = cz + max_draw_dist
        # For Z-lines, vary X similarly
        x0 = cx - max_draw_dist
        x1 = cx + max_draw_dist

        # helper to project world point and check simple behind-camera clip
        def proj_point(p):
            cam_space = cam.world_to_camera(p)
            sp = cam.camera_to_screen(cam_space)
            return sp, cam_space

        # Draw X-aligned lines (lines running along Z, varying X)
        for ix in range(ix_min, ix_max + 1):
            wx = ix * cell_size
            # quick 2D frustum cull: center of this line relative to camera
            mid_dx = wx - cx
            mid_dz = cz - cz  # zero because center z is camera cz
            # If the middle of this vertical line is significantly behind camera, skip
            if (mid_dx * fwd_x + mid_dz * fwd_z) < -cell_size * 0.5 and abs(
                mid_dx
            ) > max_draw_dist:
                continue

            # world endpoints of the line
            pA = (wx, 0.0, z0)
            pB = (wx, 0.0, z1)

            spA, camA = proj_point(pA)
            spB, camB = proj_point(pB)

            if spA is None and spB is None:
                continue

            # if at least one endpoint projects, draw line between projected points (clip by camera)
            # choose color: major vs minor (major lines every N cells)
            if (ix % major_every) == 0:
                col = major_col
                width = 2
            else:
                col = minor_col
                width = 1

            # highlight X=0 axis
            if ix == 0:
                col = axis_col
                width = 2

            # If either endpoint is None (off-screen), try to clip by intersecting in camera space:
            if spA is None or spB is None:
                # fallback: sample an intermediate point at camera z plane (simple heuristic)
                # We'll try 4 sample points along the segment and draw if any visible portion exists
                samples = 4
                pts = []
                for s in range(samples + 1):
                    t = s / samples
                    wx_s = wx
                    wz_s = z0 * (1 - t) + z1 * t
                    sp_s, _ = proj_point((wx_s, 0.0, wz_s))
                    if sp_s is not None:
                        pts.append(sp_s)
                    else:
                        pts.append(None)
                # find first and last visible sample indices
                vis_indices = [i for i, p in enumerate(pts) if p is not None]
                if not vis_indices:
                    continue
                start = pts[vis_indices[0]]
                end = pts[vis_indices[-1]]
                pygame.draw.line(self.screen, col, start, end, width)
            else:
                pygame.draw.line(self.screen, col, spA, spB, width)

        # Draw Z-aligned lines (lines running along X, varying Z)
        for iz in range(iz_min, iz_max + 1):
            wz = iz * cell_size
            mid_dx = cx - cx  # zero
            mid_dz = wz - cz
            if (mid_dx * fwd_x + mid_dz * fwd_z) < -cell_size * 0.5 and abs(
                mid_dz
            ) > max_draw_dist:
                continue

            pA = (x0, 0.0, wz)
            pB = (x1, 0.0, wz)

            spA, camA = proj_point(pA)
            spB, camB = proj_point(pB)

            if spA is None and spB is None:
                continue

            if (iz % major_every) == 0:
                col = major_col
                width = 2
            else:
                col = minor_col
                width = 1

            if iz == 0:
                col = (120, 200, 120)
                width = 2

            if spA is None or spB is None:
                samples = 4
                pts = []
                for s in range(samples + 1):
                    t = s / samples
                    wx_s = x0 * (1 - t) + x1 * t
                    wz_s = wz
                    sp_s, _ = proj_point((wx_s, 0.0, wz_s))
                    if sp_s is not None:
                        pts.append(sp_s)
                    else:
                        pts.append(None)
                vis_indices = [i for i, p in enumerate(pts) if p is not None]
                if not vis_indices:
                    continue
                start = pts[vis_indices[0]]
                end = pts[vis_indices[-1]]
                pygame.draw.line(self.screen, col, start, end, width)
            else:
                pygame.draw.line(self.screen, col, spA, spB, width)

    def draw_ground_quad(self, cam):
        # test.draw_ground(self.screen, cam)
        test.draw_ground_grid(self.screen, cam)

    def draw_crosshair(self, color=(200, 200, 200)):
        center_x, center_y = self.width // 2, self.height // 2
        size = 10

        # Draw crosshair lines
        pygame.draw.line(
            self.screen,
            color,
            (center_x - size, center_y),
            (center_x + size, center_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            color,
            (center_x, center_y - size),
            (center_x, center_y + size),
            2,
        )

    def begin_frame(self):
        self.clear()

    def draw_gui_text(self, text, pos, color=(255, 255, 255)):
        return self.draw_gui_lines([text], pos, color)

    def draw_gui_lines(self, lines, pos, color=(255, 255, 255)):
        self.fast_text_renderer.color = color
        text_surf = self.fast_text_renderer.render_lines(lines)
        self.screen.blit(text_surf, pos)

    def finish_frame(self):
        pygame.display.flip()
        self.clock.tick(FPS)


class FastTextRenderer:
    _font_cache = {}  # Cache fonts by (path, size)
    _glyph_cache = {}  # Cache rendered characters by (font_id, char, color, aa)

    def __init__(self, font_path=None, size=20, color=(255, 255, 255), aa=True):
        self.font = self.get_font(font_path, size)
        self.color = color
        self.aa = aa

        self.surface = None
        self.lines = []
        self.line_height = self.font.get_linesize()

    @classmethod
    def get_font(cls, path, size):
        key = (path, size)
        if key not in cls._font_cache:
            cls._font_cache[key] = pygame.font.SysFont(path, size)
        return cls._font_cache[key]

    @classmethod
    def render_glyph(cls, font, char, color, aa):
        key = (id(font), char, color, aa)
        if key not in cls._glyph_cache:
            cls._glyph_cache[key] = font.render(char, aa, color).convert_alpha()
        return cls._glyph_cache[key]

    def render_lines(self, lines):
        """Render list of text lines into one surface (cached glyphs)."""
        self.lines = lines

        # Render all lines as lists of glyphs
        rendered_lines = []
        max_width = 0

        for line in lines:
            glyphs = [
                self.render_glyph(self.font, ch, self.color, self.aa) for ch in line
            ]
            rendered_lines.append(glyphs)

            width = sum(g.get_width() for g in glyphs)
            max_width = max(max_width, width)

        total_height = len(lines) * self.line_height
        self.surface = pygame.Surface((max_width, total_height), pygame.SRCALPHA)

        # Blit all glyphs into block
        y = 0
        for glyphs in rendered_lines:
            x = 0
            for g in glyphs:
                self.surface.blit(g, (x, y))
                x += g.get_width()
            y += self.line_height

        return self.surface

    def draw(self, screen, x, y):
        if self.surface is not None:
            screen.blit(self.surface, (x, y))
