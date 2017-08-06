import numpy as np
import random


# The function gridWorld returns the reward function
def gridWorld():
    # Grid world layout:
    #
    #  ---------------------
    #  |  0 |  1 |  2 |  3 |
    #  ---------------------
    #  |  4 |  5 |  6 |  7 |
    #  ---------------------
    #  |  8 |  9 | 10 | 11 |
    #  ---------------------
    #  | 12 | 13 | 14 | 15 |
    #  ---------------------
    #
    #  Goal state: 15 
    #  Bad state: 9
    #  End state: 16
    #
    #  The end state is an absorbing state that the agent transitions 
    #  to after visiting the goal state.
    #
    #  There are 17 states in total (including the end state) 
    #  and 4 actions (up, down, right, left).
    #

    #%%%%%%%%%%%%%%%%%%% rewards %%%%%%%%%%%%%%%%%%%%%%

    # Rewards are stored in a one dimensional array R[s]
    #
    # All states have a reward of -1 except:
    # Goal state: 100
    # Bad state: -70
    # End state: 0 

    # initialize rewards to -1
    R = -np.ones(17)

    # set rewards
    R[15] = 100  # goal state
    R[9] = -70   # bad state
    R[16] = 0    # end state

    return R

up = 0
down = 1
left = 2
right = 3


# Grid world layout:
#
#  ---------------------
#  |  0 |  1 |  2 |  3 |
#  ---------------------
#  |  4 |  5 |  6 |  7 |
#  ---------------------
#  |  8 |  9 | 10 | 11 |
#  ---------------------
#  | 12 | 13 | 14 | 15 |
#  ---------------------

def get_new_state_fixed(init_s, direction):
    new_s = init_s
    if direction == up and init_s - 4 >= 0:
        new_s = init_s - 4
    elif direction == down and init_s + 4 < 16:
        new_s = init_s + 4
    elif direction == left and init_s not in [0, 4, 8, 12]:
        new_s = init_s - 1
    elif direction == right and init_s not in [3, 7, 11]:
        new_s = init_s + 1
    return new_s


def get_new_state(init_s, direction, b):
    if init_s == 15:
        return 16
    rand = random.random()
    if direction == left or direction == right:
        if rand < b:
            return get_new_state_fixed(init_s, up)
        if rand < 2*b:
            return get_new_state_fixed(init_s, down)
        return get_new_state_fixed(init_s, direction)
    elif direction == up or direction == down:
        if rand < b:
            return get_new_state_fixed(init_s, left)
        if rand < 2*b:
            return get_new_state_fixed(init_s, right)
        return get_new_state_fixed(init_s, direction)
    else:
        print "Wrong direction"


def pick_action(qvals, epsilon):
    if random.random() >= epsilon:
        return np.argmax(qvals)
    else:
        return random.randint(0, 3)


def q_learning(epsilon = 0.05):
    qvals = np.zeros((17, 4))
    count = np.zeros((17, 4))
    discount = 0.99
    b = 0.05
    R = gridWorld()
    actions = []
    states = []
    iterations = 10000
    for i in range(iterations):
        actions = []
        states = []
        s = 4
        while s != 16:
            states += [s]
            action = pick_action(qvals[s], epsilon)
            count[s][action] += 1
            alpha = 1.00/count[s][action]
            qs = qvals[s][action]
            actions += [action]
            new_s = get_new_state(s, action, b)
            qvals[s][action] = qs + alpha * (R[s] + discount*np.max(qvals[new_s]) - qs)
            s = new_s
    print "State Transition"
    print states
    print "--------------------------------"
    print "Actions"
    print actions
    print "--------------------------------"
    print "Q Value Table"
    print qvals

# currently epsilon set at 0.2
q_learning(0.2)


