import pygame
import pandas as pd
import numpy as np
import sys
import math
import os

def get_color(z, min_z, max_z):
    """
    Map the predicted redshift (Z) to a colormap.
    White/Yellow = Close
    Red/Purple = Far Away
    """
    norm_z = (z - min_z) / (max_z - min_z + 1e-5)
    
    if norm_z < 0.2:
        f = norm_z / 0.2
        return (int(255), int(255), int(255 * (1-f)))
    elif norm_z < 0.5:
        f = (norm_z - 0.2) / 0.3
        return (int(255), int(255 - 155*f), 0)
    elif norm_z < 0.8:
        f = (norm_z - 0.5) / 0.3
        return (int(255 - 105*f), int(100 - 100*f), int(150*f))
    else:
        f = (norm_z - 0.8) / 0.2
        return (int(150 - 100*f), 0, int(150 - 50*f))

def main(csv_file="mapped_deep_field_targets_RA161.3.csv"):
    print("==================================================")
    print("       BOOTING 3D OBSERVATORY ENGINE...           ")
    print("==================================================")
    
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    pygame.init()
    
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DEEP SPACE 3D OBSERVATORY - Live Telemetry")
    
    font = pygame.font.SysFont('Consolas', 18)
    title_font = pygame.font.SysFont('Consolas', 24, bold=True)

    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
        print(f"[SYSTEM] Loaded {len(df)} celestial objects from {csv_file}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {csv_file}. Please run process_deep_field.py first.")
        pygame.quit()
        sys.exit()

    ra = df['ra'].values
    dec = df['dec'].values
    z = df['Predicted_Redshift_Z'].values
    i_mag = df['i'].values

    # 2. Normalize Data to 3D Space (-1 to 1)
    # X = Right Ascension (Inverted for astronomy standard)
    norm_x = - (ra - np.mean(ra)) / (np.max(ra) - np.min(ra)) * 2
    # Y = Declination
    norm_y = (dec - np.mean(dec)) / (np.max(dec) - np.min(dec)) * 2
    # Z = Predicted Redshift (Depth)
    min_z, max_z = np.min(z), np.max(z)
    norm_z = (z - min_z) / (max_z - min_z) * 2 - 1

    # Sizes based on magnitude (lower is brighter)
    sizes = np.clip((26 - i_mag) * 1.5, 2, 18)

    # Generate colors based on Redshift
    colors = [get_color(z_val, min_z, max_z) for z_val in z]

    # Combine into a single 3D array
    points_3d = np.column_stack((norm_x, norm_y, norm_z))

    # 3. 3D Engine Setup
    angle_x, angle_y = 0.0, 0.0
    clock = pygame.time.Clock()
    running = True

    mouse_down = False
    last_mouse_pos = (0, 0)
    scale = 350 # Zoom level

    print("[SYSTEM] Render engine active. Switch to the Pygame window!")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    last_mouse_pos = event.pos
                elif event.button == 4: # Scroll up (zoom in)
                    scale += 25
                elif event.button == 5: # Scroll down (zoom out)
                    scale = max(50, scale - 25)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                    angle_y += dx * 0.005
                    angle_x += dy * 0.005
                    last_mouse_pos = event.pos

        # Deep space dark blue/black background
        screen.fill((5, 8, 15))

        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x), math.cos(angle_x)]
        ])
        
        rot_y = np.array([
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)]
        ])

        # Apply rotation
        rotated_points = points_3d.dot(rot_y).dot(rot_x)
        
        # Sort points by Z so further ones are drawn first (Painter's Algorithm)
        sort_indices = np.argsort(rotated_points[:, 2])

        # Draw Grid/Axes for reference
        axis_len = 1.5
        axes = np.array([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len]
        ])
        rot_axes = axes.dot(rot_y).dot(rot_x)
        
        def project(pt):
            return int(pt[0] * scale + width / 2), int(pt[1] * scale + height / 2)
            
        origin = project(rot_axes[0])
        pygame.draw.line(screen, (100, 50, 50), origin, project(rot_axes[1]), 1) # X Axis
        pygame.draw.line(screen, (50, 100, 50), origin, project(rot_axes[2]), 1) # Y Axis
        pygame.draw.line(screen, (50, 50, 100), origin, project(rot_axes[3]), 1) # Z Axis

        # Render Galaxies
        for idx in sort_indices:
            pt = rotated_points[idx]
            x_proj, y_proj = project(pt)
            
            size = sizes[idx]
            color = colors[idx]
            original_z = z[idx]
            
            # Simple depth fading (make further points slightly darker)
            fade = np.clip((pt[2] + 2) / 3, 0.3, 1.0)
            faded_color = (int(color[0]*fade), int(color[1]*fade), int(color[2]*fade))

            # Draw glowing halo
            pygame.draw.circle(screen, (faded_color[0]//3, faded_color[1]//3, faded_color[2]//3), (x_proj, y_proj), int(size * 1.5), 1)
            # Draw solid core
            pygame.draw.circle(screen, faded_color, (x_proj, y_proj), int(size))

        # Draw HUD (Heads Up Display)
        title = title_font.render("AI OBSERVATORY: 3D COSMIC DEPTH MAP", True, (0, 255, 200))
        hud_text1 = font.render("> DRAG TO ROTATE 3D SPACE", True, (150, 150, 150))
        hud_text2 = font.render("> SCROLL TO ZOOM", True, (150, 150, 150))
        hud_text3 = font.render(f"TARGETS DETECTED: {len(points_3d)}", True, (255, 255, 255))
        hud_text4 = font.render(f"Z-RANGE: {min_z:.2f} TO {max_z:.2f}", True, (255, 255, 255))
        
        screen.blit(title, (20, 20))
        screen.blit(hud_text1, (20, 60))
        screen.blit(hud_text2, (20, 85))
        screen.blit(hud_text3, (20, 120))
        screen.blit(hud_text4, (20, 145))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
