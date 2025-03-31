# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import random

# Importing the Kivy packages
from kivy.app import App
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

display_width = 2876
display_height = 1250
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
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 5, -5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
# Define goal points
goal_points = [ (100, 200), (500, 800), (1200, 300)]  # A1, A2, A3 coordinates
current_goal_index = 0  # Start with A1

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    
    sand = np.zeros((display_height, display_width))
    img = PILImage.open("./images/mask.png").convert('L')
    width, height = img.size
    sand = np.zeros((height, width)) # Create sand array with correct dimensions
    resized_img = img.resize((display_width, display_height), PILImage.Resampling.LANCZOS)
    sand = np.asarray(resized_img) / 255
    goal_x = display_width // 2
    goal_y = display_height // 2
    first_update = False
    global swap
    swap = 0

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
            self.signal1 = 10. if not (10 <= self.sensor1_x < longueur - 10 and 10 <= self.sensor1_y < largeur - 10) else int(np.sum(sand[int(self.sensor1_y) - 10:int(self.sensor1_y) + 10, int(self.sensor1_x) - 10:int(self.sensor1_x) + 10])) / 400.
            self.signal2 = 10. if not (10 <= self.sensor2_x < longueur - 10 and 10 <= self.sensor2_y < largeur - 10) else int(np.sum(sand[int(self.sensor2_y) - 10:int(self.sensor2_y) + 10, int(self.sensor2_x) - 10:int(self.sensor2_x) + 10])) / 400.
            self.signal3 = 10. if not (10 <= self.sensor3_x < longueur - 10 and 10 <= self.sensor3_y < largeur - 10) else int(np.sum(sand[int(self.sensor3_y) - 10:int(self.sensor3_y) + 10, int(self.sensor3_x) - 10:int(self.sensor3_x) + 10])) / 400.

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
        self.draw_goals()

    def draw_goals(self):
        with self.canvas.before:  # Draw under the car
            Color(1, 1, 1)  # White color
            for x, y in goal_points:
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))  # Draw a circle
    
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = goal_points[0]  # Set initial car position to A1
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global last_reward
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
            init()

        goal_x, goal_y = goal_points[current_goal_index]

        prev_x, prev_y = self.car.x, self.car.y
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        epsilon = 0.05
        if random.random() < epsilon:
            action = random.choice([0, 1, 2])
        else:
            action = brain.update(last_reward, last_signal)

        scores.append(brain.score())
        rotation = action2rotation[action] + random.uniform(-0.3, 0.3)
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        car_x_int = int(max(0, min(self.car.x, longueur - 1)))
        car_y_int = int(max(0, min(self.car.y, largeur - 1)))
        car_x_int_im = int(max(0, min(car_x_int, im.width - 1)))
        car_y_int_im = int(max(0, min(car_y_int, im.height - 1)))

        if sand[car_y_int, car_x_int] > 0:
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)
            if rotation == 0:
                last_reward = 2.5 # Increased black line reward
            else:
                last_reward = 1.5 # Increased black line reward
            print(1, goal_x, goal_y, distance, car_x_int, car_y_int, im.read_pixel(car_x_int_im, car_y_int_im))
        else:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1.0 # Reduced off-line penalty
            self.car.x, self.car.y = prev_x + random.uniform(-3, 3), prev_y + random.uniform(-3, 3) # Smaller random position change
            self.car.angle += 180 + random.uniform(-3, 3) # smaller random rotation
            self.car.velocity = Vector(-self.car.velocity_x, -self.car.velocity_y)
            print(0, goal_x, goal_y, distance, car_x_int, car_y_int, im.read_pixel(car_x_int_im, car_y_int_im))

        if self.car.x < 0 or self.car.x >= longueur or self.car.y < 0 or self.car.y >= largeur:
            last_reward = -1.5 # Reduced boundary penalty
            self.car.x, self.car.y = prev_x + random.uniform(-3, 3), prev_y + random.uniform(-3, 3) # Smaller random position change
            self.car.angle += 30 + random.uniform(-3, 3) # smaller random rotation
            self.car.velocity = Vector(-self.car.velocity_x, -self.car.velocity_y)

        if distance < 25:
            current_goal_index = (current_goal_index + 1) % len(goal_points)
            last_reward = 5.0
            print(f"New target: {goal_points[current_goal_index]}")

        last_distance = distance

    

# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(1.0, 1.0, 1.0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=8)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1  # Corrected line
            img = PILImage.fromarray(sand.astype("uint8") * 255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 1

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
        sand = np.zeros((display_width, display_height))

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