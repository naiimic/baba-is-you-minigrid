import gymnasium as gym
import pygame
from gym_minigrid.babaisyou import BabaIsYouEnv #, BabaIsYouGrid
from gym_minigrid.minigrid import MiniGridEnv #, Grid, MissionSpace,

from gym_minigrid import register_minigrid_envs
from gym.envs.registration import register
from gym_minigrid.envs.babaisyou import TestRuleEnv

from gym.utils.play import display_arr
from pygame import VIDEORESIZE

from tests.test_babaisyou import TestWinLoseEnv, TestPushEnv, TestPullEnv


def play_minigrid(env):
    assert isinstance(env.unwrapped, MiniGridEnv)
    keys_to_action = {
        (ord('q'),): env.actions.left,
        (ord('d'),): env.actions.right,
        (ord('z'),): env.actions.forward,
        (ord(' '),): env.actions.toggle
    }
    play(env, fps=30, keys_to_action=keys_to_action)


def play_babaisyou(env):
    assert isinstance(env.unwrapped, BabaIsYouEnv)
    keys_to_action = {
        (ord('z'),): env.actions.up,
        (ord('d'),): env.actions.right,
        (ord('s'),): env.actions.down,
        (ord('q'),): env.actions.left
    }
    play(env, fps=30, keys_to_action=keys_to_action)


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    env.reset()
    rendered = env.render(mode="rgb_array")
    # rendered = env.render()

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, (
                env.spec.id
                + " does not have explicit key to action mapping, "
                + "please specify one manually"
            )
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), None)  # TODO: was 0
            pressed_keys = []
            prev_obs = obs
            if action is not None:
                # obs, rew, env_done, _, info = env.step(action)
                obs, rew, env_done, info = env.step(action)
                print("Reward:", rew) if rew != 0 else None
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            # rendered = env.render()
            rendered = env.render(mode="rgb_array")
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            # elif event.type == pygame.KEYUP:
            #     if event.key in relevant_keys:
            #         pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


# Uncomment to Play Test Cases from test_babaisyou.py
# env = TestRuleEnv()
# play_babaisyou(env)

# env2 = TestWinLoseEnv()
# play_babaisyou(env2)

# env3 = TestPushEnv(collision=4)
# play_babaisyou(env3)

# env4 = TestPullEnv(collision=7)
# play_babaisyou(env4)