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