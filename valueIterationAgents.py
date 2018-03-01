# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        for _ in range(0,self.iterations):
            #declaring a temperary dictionary with the states as the key and the value iteration values as the value for the corresponding key.
            value_dict = util.Counter()
            for state in self.mdp.getStates():
                #we check if the state is terminal  state, we give the value of 0, since thats the initialised value in the dictionary.

                if mdp.isTerminal(state): value_dict[state] = 0
                else:
                    value = [float("-inf")]
                    #For each action we store the values obtained for each in a list and assign the maximum value as the value of that state.
                    for action in self.mdp.getPossibleActions(state):
                        for successor,transition in self.mdp.getTransitionStatesAndProbs(state,action):
                            #As per definition we, for a given state, we compute the Q-values for each valid action from that state and assign the maximum as the value of the state.
                            value += [self.computeQValueFromValues(state,action)]
                    value_dict[state] = max(value)
            self.values = value_dict





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        Q_value = 0
        for successor, transition in self.mdp.getTransitionStatesAndProbs(state, action):
            #Here we do the summation over the states that can be reached from the given state, by performing the given action.
            Q_value += transition * (self.mdp.getReward(state, action, successor) + (self.discount * self.values[successor]))
        return Q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #As per the given description, terminal states have zero actions.
        if self.mdp.isTerminal(state): return None
        Q_value_action_pairs = []
        for action in self.mdp.getPossibleActions(state):
            #We append a tupple of (Qvalues, action) in the list and pick the action with the maximum value as the policy.
            Q_value_action_pairs.append((self.computeQValueFromValues(state,action),action))
        policy = max(Q_value_action_pairs)[1]
        return policy


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
