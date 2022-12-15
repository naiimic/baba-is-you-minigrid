import numpy as np

from gym_minigrid.babaisyou import BabaIsYouEnv, BabaIsYouGrid
from gym_minigrid.envs.babaisyou.core.flexible_world_object import RuleObject, RuleIs, RuleProperty, Baba, make_obj, \
    FDoor


class MoveObjEnv(BabaIsYouEnv):
    def __init__(self, obj="fball", goal_pos='random', size=7, **kwargs):
        self.goal_pos = goal_pos

        obj = [obj] if not isinstance(obj, list) else obj
        self.goal_obj_name = obj
        self.goal_obj = [make_obj(o) for o in obj]

        self.size = size
        super().__init__(grid_size=self.size, max_steps=4*self.size*self.size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(RuleObject('baba'), 1, 1)
        self.put_obj(RuleIs(), 2, 1)
        self.put_obj(RuleProperty('is_agent'), 3, 1)

        idx = np.random.choice(range(len(self.goal_obj)))
        sampled_obj_name, self.sampled_obj = self.goal_obj_name[idx], self.goal_obj[idx]

        self.put_obj(RuleObject(sampled_obj_name), 1, 2)
        self.put_obj(RuleIs(), 2, 2)
        self.put_obj(RuleProperty('can_push'), 3, 2)

        if self.goal_pos == 'random':
            self.current_goal_pos = self.place_obj(FDoor())
        else:
            self.put_obj(FDoor(), *self.goal_pos)
            self.current_goal_pos = self.goal_pos
        for o in self.goal_obj:
            self.place_obj(o, top=(2, 2), size=[self.size-4, self.size-4])

        self.place_obj(Baba())
        self.place_agent()

    def reward(self):
        if self.grid.get(*self.current_goal_pos) == self.sampled_obj:
            return self._reward(), True
        else:
            return 0, False


class OpenShutObjEnv(BabaIsYouEnv):
    object_names = ["fball", "fwall", "fkey", "fdoor"]

    def __init__(self, open_objects: list[str] = None, shut_objects: list[str] = None, size=8, **kwargs):
        open_objects = self.object_names if open_objects is None else open_objects
        shut_objects = self.object_names if shut_objects is None else shut_objects
        self.open_objects = {name: make_obj(name) for name in open_objects}
        self.shut_objects = {name: self.open_objects.get(name, make_obj(name)) for name in shut_objects}
        self.all_objects = {**self.open_objects, **self.shut_objects}

        default_ruleset = {
            "is_agent": {"baba": True},
            # "can_push": {obj: True for obj in self.object_names}
        }

        self.size = size
        super().__init__(grid_size=self.size, max_steps=4*self.size*self.size, default_ruleset=default_ruleset, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # sample open and shut objects
        open_obj_name = np.random.choice(list(self.open_objects.keys()))
        # ensure that shut obj is different from open obj
        shut_objects = dict(self.shut_objects)
        if open_obj_name in shut_objects:
            del shut_objects[open_obj_name]
        shut_obj_name = np.random.choice(list(shut_objects.keys()))

        self.open_obj = self.open_objects[open_obj_name]
        self.shut_obj = self.shut_objects[shut_obj_name]

        # open and push rules
        self.put_obj(RuleObject(open_obj_name), 1, 1)
        self.put_obj(RuleIs(), 2, 1)
        self.put_obj(RuleProperty('can_push'), 3, 1)
        # self.put_obj(RuleProperty('is_open'), 3, 1)

        # self.put_obj(RuleObject(open_obj_name), 4, 1)
        # self.put_obj(RuleIs(), 5, 1)
        # self.put_obj(RuleProperty('can_push'), 6, 1)

        # shut rule
        self.put_obj(RuleObject(shut_obj_name), 1, 2)
        self.put_obj(RuleIs(), 2, 2)
        self.put_obj(RuleProperty('is_shut'), 3, 2)

        for name, obj in self.all_objects.items():
            pos = self.place_obj(obj, top=(2, 2), size=[self.size-4, self.size-4])
            if name == open_obj_name:
                self.open_obj_pos = pos
            elif name == shut_obj_name:
                self.shut_obj_pos = pos
        self.place_obj(Baba())
        self.place_agent()

    def reward(self):
        if self.grid.get(*self.shut_obj_pos) == self.open_obj:
            return self._reward(), True
        else:
            return 0, False