import pygame
import numpy as np
from debug import plot_vertices_triangles
from Camera import Camera
from renderable_object import RenderableObject
from texture import Texture, sample
from profiler import Profiler, enabled_profiler
# ========================
#  Initialization
# ========================
pygame.init()
screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True

# ========================
#  Cube Data
# ========================
cube_vertices = [
    (1.0,  1.0, -1.0),  # v0
    (1.0, -1.0, -1.0),  # v1
    (1.0,  1.0,  1.0),  # v2
    (1.0, -1.0,  1.0),  # v3
    (-1.0,  1.0, -1.0), # v4
    (-1.0, -1.0, -1.0), # v5
    (-1.0,  1.0,  1.0), # v6
    (-1.0, -1.0,  1.0), # v7
]
# The faces of the cube  three vertices each form a triangle and 2 triangles form a face of a cube
cube_faces = [
    (0, 2, 6), (0, 6, 4),   # face (0,2,6,4)
    (3, 7, 6), (3, 6, 2),   # face (3,7,6,2)
    (7, 5, 4), (7, 4, 6),   # face (7,5,4,6)
    (5, 7, 3), (5, 3, 1),   # face (5,7,3,1)
    (1, 3, 2), (1, 2, 0),   # face (1,3,2,0)
    (5, 1, 0), (5, 0, 4),   # face (5,1,0,4)
]

cube_face_colors = [
    (255, 0, 0),     # red
    (0, 255, 0),     # green
    (0, 0, 255),     # blue
    (255, 255, 0),   # yellow
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (255, 165, 0),   # orange
    (128, 0, 128),   # purple
    (165, 42, 42),   # brown
    (0, 128, 128),   # teal
    (50, 205, 50),   # lime
    (255, 192, 203)  # pink
]

vertices = np.array(cube_vertices)
triangles = np.array(cube_faces)

cam = Camera(
    position=[0, 0, 0],
    forward=[0, 0, 1],
    up=[0, 1, 0],
    fov=np.radians(60),      # 60 degree FOV
    aspect=1280/720
)

fox = RenderableObject.load_new_obj("resources/foxSitting.obj", texture_filepath="resources/colMap.bytes")
tpot =RenderableObject.load_new_obj("resources/utahTeapot.obj")
AMBIENT = 0.2
SCALE = 150

# ========================
#  Colors
# ========================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (192, 192, 192)
DARK_GRAY = (64, 64, 64)
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
YELLOW, CYAN, MAGENTA = (255, 255, 0), (0, 255, 255), (255, 0, 255)
ORANGE, PURPLE, PINK = (255, 165, 0), (128, 0, 128), (255, 192, 203)
BROWN, NAVY_BLUE, LIME_GREEN, TEAL = (165, 42, 42), (0, 0, 128), (50, 205, 50), (0, 128, 128)

# ========================
#  Math / Utility Functions
# ========================
def rotation_matrix(angle):
    
    #aapply rotation around the y axis with a given angle this rotates the vertices of the cube around the y axis
    rot_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
        #apply rotation around the x axis with a given angle this rotates the vertices of the cube around the x axis
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return rot_y @ rot_x

# Compute normal vector of triangle defined by v1,v2,v3 which tells us which way the triangle is facing
def triangle_normal(v1, v2, v3):
    u = np.array([v2[i] - v1[i] for i in range(3)])
    v = np.array([v3[i] - v1[i] for i in range(3)])
    return np.cross(u, v)
# Determine if triangle is front-facing (towards camera) based on normal's z component if if the Z
# if the normal is negative the triangle is facing the camera if the Z is positive the triangle is facing away from the camera
def is_front_facing(v1, v2, v3):
    n = triangle_normal(v1, v2, v3)
    return n[2] < 0

# ========================
#  Rendering
# ========================
def draw_cube(surface, cube_vertices, cube_faces, cube_face_colors, cube_pos, cam, scale=150):
    """
    Draw a cube on the given surface from the perspective of the camera.

    surface: pygame surface
    cube_vertices: list of 3D vertices
    cube_faces: list of triangles (indices)
    cube_face_colors: list of colors for each face
    cube_pos: np.array([x, y, z]) cube's position in world space
    cam: Camera object
    scale: scaling factor for screen coordinates
    """

    # Convert all cube vertices to camera space
    camera_vertices = []
    for v in cube_vertices:
        v_world = np.array(v) + cube_pos
        v_cam = cam.world_to_camera(v_world)
        camera_vertices.append(v_cam)

    # Project camera-space vertices to 2D screen coordinates
    projected_vertices = []
    for v_cam in camera_vertices:
        # Skip vertices behind camera
        if v_cam[2] <= 0:
            projected_vertices.append(None)  # mark as not visible
            continue

        # Perspective projection 
        x_proj = (v_cam[0] / v_cam[2]) * cam.f * cam.aspect
        y_proj = (v_cam[1] / v_cam[2]) * cam.f

        # Map to screen pixels
        screen_x = int(x_proj * scale + surface.get_width() / 2)
        screen_y = int(-y_proj * scale + surface.get_height() / 2)
        projected_vertices.append((screen_x, screen_y, v_cam[2]))

    # Compute face depth for painter's algorithm
    faces_with_depth = []
    for face_index, face in enumerate(cube_faces):
        idx1, idx2, idx3 = face
        # Skip face if any vertex is behind camera
        if (projected_vertices[idx1] is None or
            projected_vertices[idx2] is None or
            projected_vertices[idx3] is None):
            continue

        z_avg = (projected_vertices[idx1][2] +
                 projected_vertices[idx2][2] +
                 projected_vertices[idx3][2]) / 3.0
        faces_with_depth.append((z_avg, face_index, face))

    # Sort faces back-to-front
    faces_with_depth.sort(key=lambda x: x[0], reverse=True)

    # Draw visible faces
    for _, face_index, face in faces_with_depth:
        idx1, idx2, idx3 = face
        v1, v2, v3 = projected_vertices[idx1], projected_vertices[idx2], projected_vertices[idx3]
        # Simple back-face culling based on camera-space normal
        u = np.array([v2[i] - v1[i] for i in range(3)])
        v = np.array([v3[i] - v1[i] for i in range(3)])
        n = np.cross(u, v)
        if n[2] > 0:
            continue  # facing away from camera

        face_color = cube_face_colors[face_index // 2]
        pygame.draw.polygon(surface, face_color, [(v1[0], v1[1]),
                                                  (v2[0], v2[1]),
                                                  (v3[0], v3[1])])


@Profiler.timed("draw_fox")
def draw_fox(surface,fox,cam):
    scale = 150
    # Convert all cube vertices to camera space
    camera_vertices = []
    fox_vertices = []
    for v in fox.vertices:
        v_world = np.array(v) 
        fox_vertices.append(v_world)
        v_cam = cam.world_to_camera(v_world)
        camera_vertices.append(v_cam)

    # Project camera-space vertices to 2D screen coordinates
    projected_vertices = []
    for v_cam in camera_vertices:
        # Skip vertices behind camera
        if v_cam[2] <= 0:
            projected_vertices.append(None)  # mark as not visible
            continue

        # Perspective projection 
        x_proj = (v_cam[0] / v_cam[2]) * cam.f * cam.aspect
        y_proj = (v_cam[1] / v_cam[2]) * cam.f

        # Map to screen pixels
        screen_x = int(x_proj * scale + surface.get_width() / 2)
        screen_y = int(-y_proj * scale + surface.get_height() / 2)
        projected_vertices.append((screen_x, screen_y, v_cam[2]))

    # Compute face depth for painter's algorithm
    faces_with_depth = []

    
    for face_index, (face,uvface) in enumerate(zip(fox.faces, fox.uv_faces)):
        idx1, idx2, idx3 = face
        m1,m2,m3 = uvface
        c1,c2,c3 = fox.uv_coords[m1],fox.uv_coords[m2],fox.uv_coords[m3]
        average_uv =  (c1 + c2 + c3) / 3.0
        face_color = sample(fox.texture, average_uv)
        rgb = np.floor(face_color * 255).astype(int)
        face_color = tuple(rgb)
        n = triangle_normal(fox_vertices[idx1], fox_vertices[idx2], fox_vertices[idx3])
        n = n / np.linalg.norm(n)
        # Skip face if any vertex is behind camera
        if (projected_vertices[idx1] is None or
            projected_vertices[idx2] is None or
            projected_vertices[idx3] is None):
            continue

        z_avg = (projected_vertices[idx1][2] +
                 projected_vertices[idx2][2] +
                 projected_vertices[idx3][2]) / 3.0
        faces_with_depth.append((z_avg, face_index, face,face_color,n))

    # Sort faces back-to-front
    faces_with_depth.sort(key=lambda x: x[0], reverse=True)

    # Draw visible faces
    for _, face_index, face, face_color,xy in faces_with_depth:
        idx1, idx2, idx3 = face
        v1, v2, v3 = projected_vertices[idx1], projected_vertices[idx2], projected_vertices[idx3]

        

        # Average color of the triangle's texture coordinates
        
        # Simple back-face culling based on camera-space normal
        u = np.array([v2[i] - v1[i] for i in range(3)])
        v = np.array([v3[i] - v1[i] for i in range(3)])
        n = np.cross(u, v)

        if n[2] > 0:    
            continue  # facing away from camera
        sky = np.array([0,1,0])
        sky = sky 
        dotn = np.dot(xy,sky )
        brightness = max(0, dotn)
        ambient = 0.2
        shadedcolor = np.clip(np.array(face_color) * (brightness+ambient), 0, 255).astype(int)
        face_color = tuple(shadedcolor) 
        pygame.draw.polygon(surface,face_color, [(v1[0], v1[1]),
                                                  (v2[0], v2[1]),
                                                    (v3[0], v3[1])])

@Profiler.timed("draw_pot")
def draw_pot(surface,tpot,cam):
    scale = 150
    camera_vertices = []
    tpot_vertices = []
    Profiler.profile_accumulate_start("transform_vertices")
    for v in tpot.vertices:
        v_world = np.array(v) 
        v_cam = cam.world_to_camera(v_world)
        camera_vertices.append(v_cam)
        tpot_vertices.append(v_world)
    projected_vertices = []
    Profiler.profile_accumulate_end("transform_vertices")
    
    Profiler.profile_accumulate_start("project_vertices")
    for v_cam in camera_vertices:
        if v_cam[2] <= 0:
            projected_vertices.append(None)  
            continue
        x_proj = (v_cam[0] / v_cam[2]) * cam.f * cam.aspect
        y_proj = (v_cam[1] / v_cam[2]) * cam.f
        screen_x = int(x_proj * scale + surface.get_width() / 2)
        screen_y = int(-y_proj * scale + surface.get_height() / 2)
        projected_vertices.append((screen_x, screen_y, v_cam[2]))
    Profiler.profile_accumulate_end("project_vertices")
    
    
    #195 ms
    Profiler.profile_accumulate_start("compute_face_depth")
    faces_with_depth = []
    for face_index, face in enumerate(tpot.faces):
        idx1, idx2, idx3 = face
        n = triangle_normal(tpot_vertices[idx1], tpot_vertices[idx2], tpot_vertices[idx3])
        n = n / np.linalg.norm(n)
        if (projected_vertices[idx1] is None or
            projected_vertices[idx2] is None or
            projected_vertices[idx3] is None):
            continue
        z_avg = (projected_vertices[idx1][2] +
                 projected_vertices[idx2][2] +
                 projected_vertices[idx3][2]) / 3.0
        faces_with_depth.append((z_avg, face_index, face,n))
    Profiler.profile_accumulate_end("compute_face_depth")


    Profiler.profile_accumulate_start("sort_faces")
    faces_with_depth.sort(key=lambda x: x[0], reverse=True)
    Profiler.profile_accumulate_end("sort_faces")
    
    
    #225 ms
    Profiler.profile_accumulate_start("draw_faces")
    for _, face_index, face,xy in faces_with_depth:
        idx1, idx2, idx3 = face
        v1, v2, v3 = projected_vertices[idx1], projected_vertices[idx2], projected_vertices[idx3]
        u = np.array([v2[i] - v1[i] for i in range(3)])
        v = np.array([v3[i] - v1[i] for i in range(3)])
        n = np.cross(u, v)
        if n[2] > 0:
            continue 
        sky = np.array([0,1,0])
        sky = sky// np.linalg.norm(sky)
        dotn = np.dot(xy,sky )
        brightness = max(0, dotn)
        ambient = 0.2
        face_color = WHITE
        
        shadedcolor = np.clip(np.array(face_color) * (brightness+ambient), 0, 255).astype(int)
        face_color = tuple(shadedcolor)
        Profiler.profile_accumulate_end("draw_faces")
        Profiler.profile_accumulate_start("draw_polygon")
        pygame.draw.polygon(surface, face_color, [(v1[0], v1[1]),
                                                  (v2[0], v2[1]),
                                                  (v3[0], v3[1])])
        Profiler.profile_accumulate_end("draw_polygon")
    

# ========================
#  Main Loop
# ========================
angle = 0.0
rotation_speed = 1  # radians per secownd
cube_pos = np.array([0.0, 0.0, 5.0])
second_cube_pos = np.array([0, 0, 5.0])
move_speed = 2.0  # units per second
use_perspective = True
paused = False
framecount = 0
pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

new_cube_vertices = [(v[0], v[1], v[2]) for v in cube_vertices]
while running:
    dt = clock.tick(60) / 1000
    framecount +=1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:  # toggle projection
                use_perspective = not use_perspective
            elif event.key == pygame.K_SPACE:
                paused = not paused
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
                pygame.mouse.get_rel()
    keys = pygame.key.get_pressed()
    if not paused:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        screen.fill(BLACK)
       
        angle += rotation_speed * dt
        move_speed = 3.0 
        rotation_speed = 2.0

        # Enable relative mouse motion
            # Inside your main loop:
        mouse_dx, mouse_dy = pygame.mouse.get_rel()  # get how much the mouse moved this frame
        sensitivity = 0.001  # tweak to taste
        cam.rotate(mouse_dx * sensitivity, mouse_dy * sensitivity)
    
  
        #These keys allow me to move the cube around in 3D space for example if I want to move the cube left
        # I just have to shift the x position of the cube negatively and if I want to move it up I have to increase the y position
        # and if I want to move it closer to the camera I have to decrease the z position
        if keys[pygame.K_w]:
            cam.move("forward", move_speed * dt)
        if keys[pygame.K_s]:
            cam.move("backward", move_speed * dt)
        if keys[pygame.K_a]:
            cam.move("left", move_speed * dt)
        if keys[pygame.K_d]:
            cam.move("right", move_speed * dt)
        if keys[pygame.K_q]:
            cam.move("down", move_speed * dt)
        if keys[pygame.K_e]:
            cam.move("up", move_speed * dt)
        if keys[pygame.K_LEFT]:
            cam.rotate(yaw=+rotation_speed * dt, pitch=0)
        if keys[pygame.K_RIGHT]:
            cam.rotate(yaw=-rotation_speed * dt, pitch=0)
        if keys[pygame.K_UP]:
            cam.rotate(yaw=0, pitch=rotation_speed * dt)
        if keys[pygame.K_DOWN]:
            cam.rotate(yaw=0, pitch=-rotation_speed * dt)

    screen.fill(BLUE)
    if paused:
        font = pygame.font.SysFont(None, 50)
        text = font.render("PAUSED", True, (255, 255, 255))
        screen.blit(text, (200, 200))
    else:
        np.set_printoptions(precision=3, suppress=True) 
        font = pygame.font.SysFont(None, 24)  # 24px default font

        # Render the camera forward vector constantly
        forward_text = f"Camera Forward: [{cam.forward[0]:.3f}, {cam.forward[1]:.3f}, {cam.forward[2]:.3f}]"
        position_text = f"Camera Position: [{cam.position[0]:.3f}, {cam.position[1]:.3f}, {cam.position[2]:.3f}]"
        #position_text = f"Camera Position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]"
        text_surface = font.render(forward_text, True, (255, 255, 255))  # white color
        text_surface_position = font.render(position_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))  # top-left corner
        screen.blit(text_surface_position, (10, 35))  
        
        #draw_cube(screen, cube_vertices, cube_faces, cube_face_colors, cube_pos, cam)
        draw_fox(screen,fox,cam)
        #draw_pot(screen,tpot,cam)
        #draw_cube(screen,angle,cube_vertices ,second_cube_pos,use_perspective=True)  # draw orthographic version for comparison
        if framecount % 30 == 0:  # every 120 frames (~2 seconds at 60 FPS)
            Profiler.profile_accumulate_report(intervals=30)
    pygame.display.flip()

pygame.quit()
