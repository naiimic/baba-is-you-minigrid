import math

import numpy as np

from gym_minigrid.envs.babaisyou.core.utils import add_img_text
from gym_minigrid.minigrid import WorldObj, COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
from gym_minigrid.rendering import fill_coords, point_in_circle, point_in_rect, point_in_triangle, rotate_fn


properties = [
    # 'can_overlap',
    'is_block',
    'can_push',
    'is_goal',
    'is_defeat',
    'is_agent',
    'is_pull',
    'is_move'
]

objects = [
    'fball',
    'fwall',
    'fdoor',
    'baba'
]

name_mapping = {
    'fwall': 'wall',
    'fball': 'ball',
    'fdoor': 'door',
    'can_push': 'push',
    'is_block': 'stop',
    'is_goal': 'win',
    'is_defeat': 'lose',
    'is': 'is',
    'is_agent': 'you',
    'is_pull': 'pull',
    'is_move': 'move'
}
# by default, add the displayed name is the type of the object
name_mapping.update({o: o for o in objects if o not in name_mapping})


def add_object_types(object_types):
    last_idx = len(OBJECT_TO_IDX)-1
    OBJECT_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(object_types)})


def add_color_types(color_types):
    last_idx = len(COLOR_TO_IDX)-1
    COLOR_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(color_types)})


add_color_types(name_mapping.values())
add_object_types(objects)
add_object_types(['rule', 'rule_object', 'rule_is', 'rule_property'])


def make_obj(name: str):
    """
    Make an object from a string name
    """
    # TODO: make it more general
    if name == "fwall" or name == "wall":
        return FWall()
    elif name == "fball" or name == "ball":
        return FBall()
    elif name == "baba":
        return Baba()
    else:
        raise ValueError(name)


class RuleBlock(WorldObj):
    """
    By default, rule blocks can be pushed by the agent.
    """
    def __init__(self, name, type, color, can_push=True):
        super().__init__(type, color)
        self._can_push = can_push
        self.name = name = name_mapping.get(name, name)
        self.margin = 10
        img = np.zeros((96-2*self.margin, 96-2*self.margin, 3), np.uint8)
        add_img_text(img, name)
        self.img = img

    def can_overlap(self):
        return False

    def can_push(self):
        return self._can_push

    def render(self, img):
        fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), [235, 235, 235])
        img[self.margin:-self.margin, self.margin:-self.margin] = self.img

    # TODO: different encodings of the rule blocks for the agent observation
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # RuleBlock characterized by their name instead of color
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.name], 0)


class RuleObject(RuleBlock):
    def __init__(self, obj, can_push=True):
        super().__init__(obj, 'rule_object', 'purple', can_push=can_push)
        assert obj in objects, "{} not in {}".format(obj, objects)
        self.object = obj


class RuleProperty(RuleBlock):
    def __init__(self, property, can_push=True):
        super().__init__(property, 'rule_property', 'purple', can_push=can_push)
        assert property in properties, "{} not in {}".format(property, properties)
        self.property = property


class RuleIs(RuleBlock):
    def __init__(self, can_push=True):
        super().__init__('is', 'rule_is', 'purple', can_push=can_push)


def make_prop_fn(prop, typ):
    """
    Make a method that retrieves the property of a FlexibleWorldObj in the ruleset
    """
    def get_prop(self):
        ruleset = self.get_ruleset()

        # TODO: is_pull, is_agent implies is_stop
        if prop == 'is_block':
            if ruleset['is_pull'].get(typ, False) or ruleset['is_agent'].get(typ, False):
                ruleset['is_block'][typ] = True

        return ruleset[prop].get(typ, False)
    return get_prop


class FlexibleWorldObj(WorldObj):
    def __init__(self, type, color):
        assert type in objects, "{} not in {}".format(type, objects)
        super().__init__(type, color)
        # direction in which the object is facing
        self.dir = 0  # order: right, down, left, up

        for prop in properties:
            setattr(self.__class__, prop, make_prop_fn(prop, self.type))

    # TODO: might be better to use a Ruleset object
    def set_ruleset_getter(self, get_ruleset):
        self._get_ruleset = get_ruleset

    def get_ruleset(self):
        return self._get_ruleset()

    # compatibility with WorldObj
    def can_overlap(self):
        return not self.is_block()


class FWall(FlexibleWorldObj):
    def __init__(self, color="grey"):
        super().__init__("fwall", color)

    def render(self, img):
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), COLORS[self.color])


class FBall(FlexibleWorldObj):
    def __init__(self, color="green"):
        super().__init__("fball", color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class FDoor(FlexibleWorldObj):
    def __init__(self, color="red", is_open=False, is_locked=False):
        super().__init__("fdoor", color)
        self.is_open = is_open
        self.is_locked = is_locked

    # TODO
    # def toggle(self, env, pos):
    #     # If the player has the right key to open the door
    #     if self.is_locked:
    #         if isinstance(env.carrying, Key) and env.carrying.color == self.color:
    #             self.is_locked = False
    #             self.is_open = True
    #             return True
    #         return False
    #
    #     self.is_open = not self.is_open
    #     return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        # if door is closed and unlocked
        elif not self.is_open:
            state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Baba(FlexibleWorldObj):
    def __init__(self, color="white"):
        super().__init__("baba", color)

    def render(self, img):
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, (255, 255, 255))
