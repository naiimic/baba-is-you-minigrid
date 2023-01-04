from gym.envs.registration import register
from gym_minigrid.minigrid import Wall

def register_minigrid_envs():
    # register BabaIsYou envs
    register(
        id="BabaIsYou-GoToObj-v0",
        entry_point="gym_minigrid.envs:GoToObjEnv"
    )
    register(
        id="BabaIsYou-GoToWinObj-v0",
        entry_point="gym_minigrid.envs:GoToWinObjEnv"
    )
    register(
        id="BabaIsYou-ChangeRule-v0",
        entry_point="gym_minigrid.envs:ChangeRuleEnv"
    )
    register(
        id="BabaIsYou-TestRule-v0",
        entry_point="gym_minigrid.envs:TestRuleEnv"
    )
    register(
        id="BabaIsYou-MoveBlock-v0",
        entry_point="gym_minigrid.envs:MoveBlockEnv"
    )
    register(id="BabaIsYou-MoveObj-v0",
        entry_point="gym_minigrid.envs:MoveObjEnv"
    )
    register(id="BabaIsYou-OpenShutObj-v0",
        entry_point="gym_minigrid.envs:OpenShutObjEnv"
    )
    register(
        id="BabaIsYou-MakeRule-v0",
        entry_point="gym_minigrid.envs:MakeRuleEnv"
    )
    register(
        id="BabaIsYou-Test-v0",
        entry_point="gym_minigrid.envs:TestEnv"
    )
