import numpy as np
import time


class Camera:
    def __init__(self, position, forward, up, fov, aspect):
        self.position = np.array(position, dtype=float)
        #The direction the camera is facing example [0,0,-1] means looking down the negative z axis
        self.forward = self._normalize(np.array(forward, dtype=float))
        # a vector that points upwards from the camera's perspective, used to determine the camera's vertical orientation
        self.up = self._normalize(np.array(up, dtype=float))
        # A vector perpendicular to both the forward and up vectors, representing the camera's right direction
        #used to determine where strafe movement goes
        self.right = self._normalize(np.cross(self.forward, self.up))
        
        self.yaw = 0  # Horizontal rotation angle (left/right)
        self.pitch = 0         # Vertical rotation angle (up/down)
        self.update_vectors()
        # Perspective projection
            
            # f value determines the scaling based on fov, bigger fov means biggger f values which means objects appear smaller
            # we divide by f later so bigger f means smaller projected values
            # for fun you can change the near plane to see how it affects the projection
            # doubling near  double the f and when later we divide by f the projected values become doubled on screen
        near=1.0
            # The bigger the fov angle means the bigger the fustrum wall becomes at the near plane
            # The points occopy less of the screen when fov is larger because the x and y values are divided by a larger f value
            # making the cube appear smaller
            
            

            
            # I scale by z to create the perspective effect of things farther away appearing smaller like in the real world otherwise
            # I would be using an Orthographic projection which ignores distance and just uses the x and y values directly
            # I also adjust for aspect ratio to avoid distortion and make sure to scale by f to account for the field of view
            # A smaller fov means a larger f value which means more zoomed out which means that I see less of the world as
            # Smaller FOV → smaller f → projected values are larger → objects appear bigger (zoomed in).
            # nearer objects take up more of the screen
        
        self.fov = fov
        self.aspect = aspect
        self.f =  1 *np.tan(fov / 2)*2
        self._last_debug_print = time.time()
        self._debug_interval = 8.0  # seconds


    
    # this normalizes the vector so that it has a length of 1 but keeps its direction the same
    # so if our vector is (3,4,0) it will become (0.6,0.8,0) it now has a maximum length of 1 but points in the same direction
    # if we don't normalize we will get weird results when we move the camera where we move faster the the longer the vector is aka the further we move from the origin
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

# Camera.py - RECOMMENDED FPS-STYLE move METHOD

    def move(self, direction, amount):
        """
        Moves the camera (FPS style: W/S are always horizontal).
        """
        world_up = np.array([0, 1, 0])

        if direction == "forward":
            # Use the flattened vector to ensure horizontal movement only
            self.position += self.forward_horizontal() * amount
            
        elif direction == "backward":
            # Use the flattened vector to ensure horizontal movement only
            self.position -= self.forward_horizontal() * amount
        elif direction == "right":
            # Strafing remains local to the camera's orientation
            self.position += self.right * amount
        elif direction == "left":
            self.position -= self.right * amount
        elif direction == "up":
            self.position += world_up * amount
        elif direction == "down":
            self.position -= world_up * amount
        #self.debug_print()


    def rotate(self, yaw_delta, pitch_delta):
        self.yaw += yaw_delta
        self.pitch += pitch_delta
        
        # clamp pitch to avoid looking straight up/down
        self.pitch = np.clip(self.pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)

        self.update_vectors()

    def update_vectors(self):
        # Compute forward from yaw/pitch
        fx = np.cos(self.pitch) * np.cos(self.yaw)
        fy = np.sin(self.pitch)
        fz = np.cos(self.pitch) * np.sin(self.yaw)
        self.forward = self._normalize(np.array([fx, fy, fz]))

        # Right vector
        world_up = np.array([0,1,0])
        self.right = self._normalize(np.cross(self.forward, world_up))

        # Up vector
        self.up = np.cross(self.right, self.forward)


    # Returns the forward vector projected onto the horizontal plane (y=0) removes any vertical component
    # imagine your looking up slightly this flattens is you just look in the direction along the ground
    def forward_horizontal(self):
        """Forward vector projected onto horizontal plane (y=0)"""
        f = self.forward.copy()
        f[1] = 0
        return self._normalize(f)

    # converts a point from world space to camera space
    # we need to convert points to camera space so that we can then project them onto the 2D screen
    #Camera relative position to itself is always (0,0,0)
    def world_to_camera(self, vertex):
        v = np.array(vertex) - self.position
        # we need to rotate the objects in the world so that they are relative to the camera's orientation
        # we do this by creating a rotation matrix from the camera's right, up, and
        rot_matrix = np.array([self.right, self.up, self.forward]).T
        return rot_matrix @ v

    def project_to_screen(self, vertex):
        """
        Convert a world-space vertex to 2D screen coordinates using perspective projection
        """
        # 1. Convert vertex to camera space
        v_cam = self.world_to_camera(vertex)  # [x, y, z] in camera space

        # 2. Prevent division by zero
        if v_cam[2] == 0:
            v_cam[2] = 1e-6

        # 3. Perspective projection using focal length (self.f)
        x_2d = (v_cam[0] / v_cam[2]) * self.f * self.aspect
        y_2d = (v_cam[1] / v_cam[2]) * self.f

        return np.array([x_2d, y_2d])

    def debug_print(self):
        np.set_printoptions(precision=3, suppress=True) 
        current_time = time.time()
        if current_time - self._last_debug_print >= self._debug_interval:
            print("########### CAMERA DEBUG ###########\n\n")
            print(f"Camera forward vector:{self.forward}")
          #  print("#################################")
           # print("Camera right vector  :", self.right)
           # print("#################################")
           # print("Camera up vector     :", self.up)
           # print("#################################")
            self._last_debug_print = current_time
            print("#################################")
            print("Camera position      :", self.position)
  