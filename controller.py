import pygfx as gfx
import math

class SurgicalController(gfx.controllers.Controller):
    """
    A custom camera controller that translates keyboard inputs
    into camera panning, orbiting, or zooming.
    """
    def __init__(self, camera, H, W,  **kwargs):
        super().__init__(camera, **kwargs)

        self.radius = 250
        self.height = H
        self.width = W
        # You can define customized speeds here
        self.rotation_speed = 2
        self.radius_speed = 5
        self.fixed_view = 0
        self.viewing_directions = [0, 90, 180, 270]
        self.current_angle = 0
        self.elevation = 0
        self.active_keys = {
            'fix_ccw' : False,
            'fix_cw' : False,
            'rot_ccw' : False,
            'rot_cw' : False
        }
       

    def handle_event(self, event, *args):
        # Always check if the controller is enabled
        
        if not self.enabled:
            return

        # We are only interested in key down events
        if event.type == "key_down":
            key = event.key

            if key.lower() == 'h':  
                self.active_keys['fix_ccw'] = True
            elif key == ' ':
                self.active_keys['fix_cw'] = True
            elif key.lower() == 'a':
                self.active_keys['rot_ccw'] = True
            elif key.lower() == 'd':
                self.active_keys['rot_cw'] = True
            elif key.lower() == 'w':
                self.active_keys['inc_el'] = True
            elif key.lower() == 's':
                self.active_keys['dec_el'] = True
            elif key.lower() == 'q':
                self.active_keys['zoom_in'] = True
            elif key.lower() == 'e':
                self.active_keys['zoom_out'] = True

        if event.type == "key_up":
            key = event.key
            if key.lower() == 'a':
                self.active_keys['rot_ccw'] = False
            elif key.lower() == 'd':
                self.active_keys['rot_cw'] = False
            elif key.lower() == 'w':
                self.active_keys['inc_el'] = False
            elif key.lower() == 's':
                self.active_keys['dec_el'] = False
            elif key.lower() == 'q':
                self.active_keys['zoom_in'] = False
            elif key.lower() == 'e':
                self.active_keys['zoom_out'] = False

    def tick(self):
        """
        Executed EVERY frame by the pygfx engine if auto_update is True.
        Handles both linear movement and dynamic orbital keypress rotation.
        """
        # 1. Call the parent class tick logic (handles native damping/interpolation)
        changed_states = super().tick()
        # Pull the current camera configuration dictionary
        moved = False
        snap_into_place = False
        print(self.cameras[0].local.position)
        # --- 3. Dynamic Orbit Rotation via Keys ('h', 'space', 'a', 'd') ---
        # Handle key input flags mapped within your event system
        if self.active_keys.get("fix_ccw"):   # 'h' key
            self.fixed_view = (math.floor(self.current_angle / 360 * 4) + 1) % len(self.viewing_directions)
            self.current_angle = self.viewing_directions[self.fixed_view]
            moved = True
            snap_into_place = True
            self.active_keys["fix_ccw"] = False
        elif self.active_keys.get("fix_cw"): # Space key
            self.fixed_view = (math.ceil(self.current_angle / 360 * 4) - 1) % len(self.viewing_directions)
            self.current_angle = self.viewing_directions[self.fixed_view]
            moved = True
            snap_into_place = True
            self.active_keys["fix_cw"] = False
        elif self.active_keys.get("rot_ccw"):       # 'a' key
            print("Rotate ccw")
            self.current_angle += self.rotation_speed
            moved = True
        elif self.active_keys.get("rot_cw"):      # 'd' key
            print("Rotate cw")
            self.current_angle -= self.rotation_speed
            moved = True
        elif self.active_keys.get("inc_el"):       # 'a' key
            old_el = self.elevation
            self.elevation = min(self.elevation + self.rotation_speed, 75)
            moved = old_el != self.elevation
        elif self.active_keys.get("dec_el"):      # 'd' key
            old_el = self.elevation
            self.elevation = max(self.elevation - self.rotation_speed, -75)
            moved = old_el != self.elevation
        elif self.active_keys.get("zoom_in"):       # 'a' key
            self.radius = max(self.radius - self.radius_speed, 50) 
            moved = True
        elif self.active_keys.get("zoom_out"):      # 'd' key
            self.radius = min(self.radius + self.radius_speed, 500)
            moved = True

        print(self.radius)
        if moved and self.cameras:
            cam = self.cameras[0]  # Grab the tracked camera object
            # 1. Turn your dynamic angle into a 3D unit direction vector
            azimuth_rad = math.radians(self.current_angle)
            elevation_rad = math.radians(self.elevation)
            x = -math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = -math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = -math.sin(elevation_rad)
            view_dir = (x, y, z)

            # 2. Force show_object directly on the camera matrix topology
            cam.show_object(
                (0, 0, 0, self.radius),
                view_dir=view_dir,
                up=(0, 0, 1)
            )

            # 3. Tell pygfx a layout redraw step is required
 