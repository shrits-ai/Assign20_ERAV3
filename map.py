# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import random

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.image import Image as KivyImage # Add this import
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

display_width = 1429
display_height = 660
# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width', display_width)
Config.set('graphics', 'height', display_height)

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(9, 3, 0.9)
action2rotation = [0, 5, -5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
# Define goal points
goal_points = [(100, 200), (500, 400), (1200, 300)]  # A1, A2, A3 coordinates
current_goal_index = 0  # Start with A1

# Initializing the map
first_update = True


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global display_width, display_height # Ensure these are accessible

    # Load the mask image
    img = PILImage.open("./images/mask.png").convert('L')

    # --- NEW: Rotate the image ---
    # Rotate 90 degrees anti-clockwise to match citymap.png
    # PIL's rotate is anti-clockwise. expand=True adjusts canvas size for non-square images.
    img = img.rotate(90, expand=True)
    # ---------------------------

    # Resize the *rotated* image to fit the display dimensions
    # Note: display_width and display_height should match the dimensions of the rotated map (citymap.png)
    resized_img = img.resize((display_width, display_height), PILImage.Resampling.LANCZOS)

    # Convert to numpy array and normalize (black path -> 0.0, white background -> 1.0)
    sand = np.asarray(resized_img) / 255.0 # Use 255.0 for float division

    # Flip vertically to align coordinate systems (PIL top-left vs Kivy bottom-left)
    sand = np.flipud(sand)

    # Set initial goal (can remain the same, coordinates are relative to the window)
    goal_x = display_width // 2
    goal_y = display_height // 2
    first_update = False
    # global swap # swap seems unused, consider removing if not needed elsewhere
    # swap = 0


# Initializing the last distance
last_distance = 0


# Creating the car class
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        global longueur, largeur
        longueur = display_width
        largeur = display_height
        self.pos = Vector(*self.velocity) + self.pos
        # Clamp car position
        self.x = max(0, min(self.x, longueur - 1))
        self.y = max(0, min(self.y, largeur - 1))
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        # Corrected sensor signal calculation (swapped x and y)
        signal1_value = 10. if not (
                    10 <= self.sensor1_x < longueur - 10 and 10 <= self.sensor1_y < largeur - 10) else float(np.mean(
            sand[max(0, int(self.sensor1_y) - 10):min(largeur, int(self.sensor1_y) + 10),
                max(0, int(self.sensor1_x) - 10):min(longueur, int(self.sensor1_x) + 10)]))
        signal2_value = 10. if not (
                    10 <= self.sensor2_x < longueur - 10 and 10 <= self.sensor2_y < largeur - 10) else float(np.mean(
            sand[max(0, int(self.sensor2_y) - 10):min(largeur, int(self.sensor2_y) + 10),
                max(0, int(self.sensor2_x) - 10):min(longueur, int(self.sensor2_x) + 10)]))
        signal3_value = 10. if not (
                    10 <= self.sensor3_x < longueur - 10 and 10 <= self.sensor3_y < largeur - 10) else float(np.mean(
            sand[max(0, int(self.sensor3_y) - 10):min(largeur, int(self.sensor3_y) + 10),
                max(0, int(self.sensor3_x) - 10):min(longueur, int(self.sensor3_x) + 10)]))

        self.signal1 = signal1_value
        self.signal2 = signal2_value
        self.signal3 = signal3_value

class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Creating the game class
class Game(Widget):
    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        # REMOVED the canvas.before block that added self.bg = KivyImage(...)
        # REMOVED self.bind(size=self._update_bg_size)
        self.draw_goals() # Keep your goal drawing

    def draw_goals(self):
        # NOTE: This also uses canvas.before. Ensure your .kv rule uses canvas.before too,
        # or adjust where goals/background are drawn (e.g., canvas vs canvas.before/after)
        # if layering becomes an issue. For now, keeping this as is.
        with self.canvas.before:  # Draw goals *before* other Game canvas elements
            Color(1, 1, 1)  # White color
            for x, y in goal_points:
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))  # Draw a circle

    # REMOVED the _update_bg_size method entirely

    # Keep the ObjectProperty definitions
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        # Make sure the ObjectProperties are linked correctly in your kv file (looks like they are)
        self.car.pos = goal_points[0]  # Set initial car position to A1
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        # Keep the entire update method as it was in the previous corrected version
        # ... (all the logic for state, action, move, reward calculation, boundary check, goal check) ...
        # --- Step 1: Get the current state ---
        global brain
        global last_reward # Make sure this tracks the reward from the *previous* step
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global current_goal_index

        longueur = display_width
        largeur = display_height
        if first_update:
            init() # Assuming init() is correctly setting up the 'sand' array

        goal_x, goal_y = goal_points[current_goal_index]

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0

        # --- Calculate distances to boundaries (normalized) ---
        dist_left = self.car.x / longueur
        dist_right = (longueur - self.car.x) / longueur
        dist_bottom = self.car.y / largeur
        dist_top = (largeur - self.car.y) / largeur
        # ----------------------------------------------------

        # --- Original state ---
        # current_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        # --- New state including boundary distances ---
        current_signal = [self.car.signal1, self.car.signal2, self.car.signal3,
                        orientation, -orientation,
                        dist_left, dist_right, dist_bottom, dist_top] # Now 9 elements
        # --- Step 2: Update brain and select next action ---
        action = brain.update(last_reward, current_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]

        # --- Step 3: Move the car ---
        prev_x, prev_y = self.car.x, self.car.y
        self.car.move(rotation)

        # --- Step 4: Calculate reward for the action just taken ---
        new_last_reward = 0
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        car_x_int = int(max(0, min(self.car.x, longueur - 1)))
        car_y_int = int(max(0, min(self.car.y, largeur - 1)))

        on_path = False
        if 0 <= car_y_int < largeur and 0 <= car_x_int < longueur:
            if sand[car_y_int, car_x_int] < 0.1: # Adjust threshold if needed
                on_path = True

        if on_path:
            new_last_reward = 1.0
            path_centering_reward = (1.0 - self.car.signal1) + (1.0 - self.car.signal2) + (1.0 - self.car.signal3)
            new_last_reward += path_centering_reward * 0.5
            if distance < last_distance: new_last_reward += 0.5
            else: new_last_reward -= 0.2
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        else: # Off path
            new_last_reward = -5.0
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)

        # Check for hitting boundaries (using the new position)
        if not (10 <= self.car.x < longueur - 10 and 10 <= self.car.y < largeur - 10):
            # Apply boundary correction and penalty
            self.car.velocity = Vector(1, 0).rotate(self.car.angle + 180) # Reverse velocity direction
            new_last_reward = -50.0 # Heavy penalty
            #print("Out of bounds, corrective action, clamping position.")

            # --- CLAMPING LOGIC ---
            # Force the car's position to be just inside the boundary it exceeded.
            if self.car.x < 10:
                self.car.x = 10 # Clamp to left boundary
                # Optional: Consider stopping horizontal movement instantly if needed
                # self.car.velocity_x = 0
            elif self.car.x >= longueur - 10:
                self.car.x = longueur - 11 # Clamp to right boundary (just inside)
                # Optional: Consider stopping horizontal movement instantly if needed
                # self.car.velocity_x = 0

            if self.car.y < 10:
                self.car.y = 10 # Clamp to bottom boundary
                # Optional: Consider stopping vertical movement instantly if needed
                # self.car.velocity_y = 0
            elif self.car.y >= largeur - 10:
                self.car.y = largeur - 11 # Clamp to top boundary (just inside)
                # Optional: Consider stopping vertical movement instantly if needed
                # self.car.velocity_y = 0

        if distance < 25: # Reached goal
            current_goal_index = (current_goal_index + 1) % len(goal_points)
            new_last_reward = 20.0
            print(f"Goal Reached! New target: {goal_points[current_goal_index]}")

        # --- Step 5: Store results for the next iteration ---
        last_reward = new_last_reward
        last_distance = distance

        # Update sensor ball positions
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3


        # Print statements for debugging (optional)
        print(f"On Path: {on_path}, Reward: {last_reward:.2f}, Distance: {distance:.1f}, Goal: {goal_points[current_goal_index]}")
        print(f"Car Position: ({self.car.x:.1f}, {self.car.y:.1f}), Angle: {self.car.angle:.1f}")
        print(f"Signals: S1={self.car.signal1:.2f}, S2={self.car.signal2:.2f}, S3={self.car.signal3:.2f}")
# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=16)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            y_clamped = min(int(touch.y), display_height - 1)
            sand[y_clamped, int(touch.x)] = 1
            img = PILImage.fromarray(sand.astype("uint8") * 255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            y_clamped_min = max(0, int(touch.y) - 10)
            y_clamped_max = min(display_height, int(touch.y) + 10)
            sand[y_clamped_min: y_clamped_max, int(touch.x) - 10: int(touch.x) + 10] = 1

            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)
class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((display_height, display_width))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()