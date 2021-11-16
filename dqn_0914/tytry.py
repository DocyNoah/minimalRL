from common.utils import *
import gym


def main():
    env = gym.make('LunarLander-v2')

    printv(env.action_space)
    # https://gym.openai.com/envs/LunarLander-v2/
    '''
    action(4)
        no-op
        fire left engine
        fire main engine
        fire right engine 
    '''

    printv(env.observation_space.shape)
    # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L350
    '''
    observation(8)
        pos_x
        pos_y
        vel_x
        vel_y
        angle
        angularVelocity
        left_leg_ground_contact
        right_leg_ground_contact
    '''


if __name__ == "__main__":
    main()
