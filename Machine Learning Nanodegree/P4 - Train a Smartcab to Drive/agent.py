import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.QLearning = {}         # List of Q values for a (state,action)
        self.initialQ = 0           # The initial value for all the (state,action)
        self.learningRate = 0.8     # Learning Rate as shown in the Udacity Lesson
        self.discountRate = 0.3     # Discount Rate as shown in the Udacity Lesson
        self.epsilon = 100          # How to choose when to perform a random action. After 100 trial the agent should choose actions exclusively from the policy
        self.firstTry = True        # In the first timestep of each trial, the prevState, prevAction and prevReward are all null so we can't update the policy.
        self.prevState = []         # Variable to update the policy
        self.prevAction = None      # Variable to update the policy
        self.prevReward = 0         # Variable to update the policy

        # Metrics to asses the performance after 100 trials.
        self.trials = 0
        self.timeStep = 0
        self.trialDeadline = 0
        self.onTime = 0
        self.penalties = 0
        self.sumTimeSteps = 0
        self.averageTimeSteps = 0



    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.firstTry = True
        self.prevState = []
        self.prevAction = None
        self.prevReward = 0


        if self.trials > 100:
            if self.timeStep <= self.trialDeadline:
                self.onTime += 1
            self.sumTimeSteps += self.timeStep
            self.averageTimeSteps = self.sumTimeSteps/(self.trials-100)

        self.trialDeadline = self.env.get_deadline(self)
        self.trials += 1
        self.timeStep = 0

        print self.QLearning

        if self.trials > 100:
            print "New trial following only the policy: Trial = {}, On Time = {}, Penalties = {}, Average Time Steps = {}".format(self.trials-100,self.onTime,self.penalties,self.averageTimeSteps)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.timeStep += 1

        # TODO: Update state
        self.state = tuple([('waypoint',self.next_waypoint),('light',inputs['light']),('oncoming',inputs['oncoming']),('right',inputs['right']),('left',inputs['left'])])

        # TODO: Select action according to your policy
        # action = random_actions() # Take a random action without considering the enviroment
        # action = random_env_actions(inputs['light'],inputs['oncoming']) # Take a random action considering the intersection state (traffic light and oncoming traffic)

        # Choose to perform a random action or an action according to the policy.
        # As the number of trials get closer to 100, the agent should choose more actions from the policy than at random.
        # From trial 100 forward, the agent chooses only actions from the policy.
        if random.randint(1,100)<=(self.epsilon-self.trials):
            action = random_actions()
        else:
            best_q = -1000
            best_action = None
            # Check which of the valid actions is the best one to perform according to the policy.
            for validActions in Environment.valid_actions:
                q = self.QLearning.get((self.state,validActions),self.initialQ)
                if q > best_q:
                    best_q = q
                    best_action = validActions
            action = best_action

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward < 0 and self.trials>100:
            self.penalties += 1

        # TODO: Learn policy based on state, action, reward

        # I'm going to update the policy based on the state, action and reward from the previous and current time step.
        # In this case, we can't update the policy at the first time step of each trial
        if not self.firstTry:
            self.QLearning[(self.prevState,self.prevAction)] = (1 - self.learningRate) * self.QLearning.get((self.prevState,self.prevAction),self.initialQ) \
                                                             + self.learningRate * (self.prevReward + self.discountRate * self.QLearning.get((self.state,action),self.initialQ))
        else:
            self.firstTry = False

        self.prevState = self.state
        self.prevReward = reward
        self.prevAction = action

        print "LearningAgent.update(): deadline = {}, next_waypoint = {}, inputs = {}, action = {}, reward = {}".format(deadline,self.next_waypoint, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.000000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=1100)  # press Esc or close pygame window to quit


def random_actions():
    return random.choice(Environment.valid_actions)

def random_env_actions(light,oncoming):
    if light == 'red':
        return random.choice(['right',None])
    elif oncoming == 'forward':
        return random.choice(['forward','right'])
    else:
        return random.choice(['forward','left','right'])


if __name__ == '__main__':
    run()
