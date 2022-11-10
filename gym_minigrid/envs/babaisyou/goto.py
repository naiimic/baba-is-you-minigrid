from __future__ import annotations

from typing import Tuple

import numpy as np

from .core.flexible_world_object import FBall, FWall, RuleProperty, RuleIs, RuleObject, Baba
from .core.utils import grid_random_position
from gym_minigrid.minigrid import MiniGridEnv, MissionSpace, Grid
from ...babaisyou import BabaIsYouGrid, BabaIsYouEnv

RuleObjPos = Tuple[int, int]
RuleIsPos = Tuple[int, int]
RulePropPos = Tuple[int, int]


class BaseGridEnv(BabaIsYouEnv):
    def __init__(self, size, **kwargs):
        self.size = size
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )

    def put_rule(self, obj: str, property: str, positions: list[RuleObjPos, RuleIsPos, RulePropPos], can_push=True):
        self.put_obj(RuleObject(obj, can_push=can_push), *positions[0])
        self.put_obj(RuleIs(can_push=can_push), *positions[1])
        self.put_obj(RuleProperty(property, can_push=can_push), *positions[2])


def random_rule_pos(size, margin):
    rule_pos = grid_random_position(size, n_samples=1, margin=margin)[0]
    rule_pos = [(rule_pos[0]-1, rule_pos[1]), rule_pos, (rule_pos[0]+1, rule_pos[1])]
    return rule_pos


class GoToObjEnv(BaseGridEnv):
    # OBJECT_TO_IDX = {
    #     "empty": 0,
    #     "wall": 1,
    #     "fball": 2,
    #     "baba": 3
    # }
    # unencoded_object = {
    #     "rule_object": 1,
    #     "rule_is": 1,
    #     "rule_property": 1
    # }
    # COLOR_TO_IDX = {}
    # STATE_TO_IDX = {}

    def __init__(self, size=8, agent_start_dir=0, rdm_rule_pos=False, rdm_ball_pos=False, rdm_agent_pos=False,
                 push_rule_block=False, n_balls=1, **kwargs):
        self.size = size
        self.agent_start_dir = agent_start_dir
        self.rdm_rule_pos = rdm_rule_pos
        self.rdm_ball_pos = rdm_ball_pos
        self.rdm_agent_pos = rdm_agent_pos
        self.push_rule_block = push_rule_block
        self.n_balls = n_balls
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        # rule blocks position
        if self.rdm_rule_pos:
            # self.rule_pos = random_position(self.size, n_samples=3, margin=3)
            self.rule_pos = random_rule_pos(self.size, margin=2)
        else:
            # self.rule_pos = [(2, 2), (3, 2), (4, 2)]
            self.rule_pos = [(1, 2), (2, 2), (3, 2)]

        # agent and ball positions
        agent_start_pos, self.ball_pos = grid_random_position(self.size, n_samples=2, margin=1)
        while agent_start_pos in self.rule_pos or self.ball_pos in self.rule_pos:
            agent_start_pos, self.ball_pos = grid_random_position(self.size, n_samples=2, margin=1)

        self.agent_start_pos = agent_start_pos

        # Create an empty grid
        self.grid = BabaIsYouGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # self.put_obj(FBall(), *self.ball_pos)

        self.put_rule(obj='baba', property='is_agent', positions=[(1, 1), (2, 1), (3, 1)])
        self.put_rule(obj='fball', property='is_goal', positions=self.rule_pos, can_push=self.push_rule_block)

        if self.rdm_ball_pos:
            for i in range(self.n_balls):
                self.place_obj(FBall())
        else:
            self.put_obj(FBall(), 4, 4)

        if self.rdm_agent_pos:
            self.place_obj(Baba())
        else:
            self.put_obj(Baba(), 2, 5)
        self.place_agent()

    # def gen_obs(self):
    #     array = np.zeros((self.grid.width, self.grid.height, 3), dtype="uint8")
    #
    #     for i in range(self.grid.width):
    #         for j in range(self.grid.height):
    #             v = self.grid.get(i, j)
    #
    #             if v is None:
    #                 array[i, j, 0] = self.OBJECT_TO_IDX["empty"]
    #                 array[i, j, 1] = 0
    #                 array[i, j, 2] = 0
    #
    #             else:
    #                 if v.type in self.OBJECT_TO_IDX:
    #                     idx = self.OBJECT_TO_IDX[v.type]
    #                 else:
    #                     idx = self.unencoded_object[v.type]
    #                 array[i, j, :] = [idx, 0, 0]
    #
    #     return array


class GoToWinObjEnv(BaseGridEnv):
    def __init__(self, size=6, rdm_pos=False, **kwargs):
        self.rdm_pos = rdm_pos
        super().__init__(size=size, **kwargs)

    def encode_rules(self, mode='matrix'):
        ruleset = self.get_ruleset()
        objects = {'fball': 0, 'fwall': 1, 'baba': 2}
        properties = {'is_goal': 0, 'is_defeat': 1, 'is_agent': 2}

        rule_encoding = np.zeros((len(objects), len(properties)))
        rules = []
        for property in ruleset.keys():
            for obj in ruleset[property]:
                if ruleset[property][obj]:
                    rule_encoding[objects[obj], properties[property]] = 1
                    rules.append((objects[obj], properties[property]))

        if mode == 'matrix':
            return rule_encoding
        elif mode == 'list':
            return rules
        else:
            raise ValueError

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # randomly sample which object is win
        is_ball_win = np.random.choice([0, 1])

        self.rule1_pos = [(1, 1), (2, 1), (3, 1)]
        self.rule2_pos = [(1, 2), (2, 2), (3, 2)]

        ball_property = 'is_goal' if is_ball_win else 'is_defeat'
        wall_property = 'is_defeat' if is_ball_win else 'is_goal'
        self.put_rule('fball', ball_property, self.rule1_pos)
        self.put_rule('fwall', wall_property, self.rule2_pos)
        self.put_rule(obj='baba', property='is_agent', positions=[(1, 3), (2, 3), (3, 3)])

        # wall_pos, ball_pos = grid_random_position(self.size, n_samples=2, margin=1,
        #                                           exclude_pos=[*self.rule1_pos, *self.rule2_pos])

        if not self.rdm_pos:
            wall_pos = (1, 4)
            ball_pos = (3, 4)
            baba_pos = (2, 4)
            self.put_obj(FWall(), *wall_pos)
            self.put_obj(FBall(), *ball_pos)
            self.put_obj(Baba(), *baba_pos)
        else:
            self.place_obj(FWall())
            self.place_obj(FBall())
            self.place_obj(Baba())

        # self.agent_pos = (2, 4)
        # self.agent_dir = 0
        self.place_agent()
