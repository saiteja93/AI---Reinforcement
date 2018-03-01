# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        #Default dictionary to store the Q values. (state,action) as key and the Q_value is the value.
        self.Q_Values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        #We need to return 0 if the (state,action) pair arises for the first time.
        if (state,action) not in self.Q_Values:
            self.Q_Values[(state,action)] = 0
        return self.Q_Values[(state,action)]



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        #Value at a state is the maximum value amoung all Q(state,actions) values.
        Q_values = [self.getQValue(state,action) for action in self.getLegalActions(state)]
        if Q_values: return max(Q_values)
        #If terminal state, it is given a value 0
        else: return 0.0



    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        #As per the requirements, returning None if there are none Legal Actions from that state.
        if len(self.getLegalActions(state))  == 0: return None
        else:
            #we make a list of tuples, of the (Qvalues, action) and return action with maximum Qvalues.
            temp = []
            for action in self.getLegalActions(state):
                temp.append((self.getQValue(state,action),action))
            best = max(temp)[1]
            return best

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        #Probability self.epsilon, take a random action from a list of legalactions and (1-self.epsilon)take the best policy action.
        if legalActions:
            #We use the above mentioned functions, util.flipcoin and random.choice to get the job done.
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        #As taught in class, we use the Q-learning formula to update Q(s,a) for a given state s and an action a.
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.Q_Values[(state,action)] = (1-self.alpha)*self.getQValue(state,action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        #We form a dictionary with features as key, and their corresponding values as their values.
        features_dictionary = self.featExtractor.getFeatures(state,action)
        #Here we perform the dot product of the value of the features and their corresponding weights.
        return sum(self.weights[feat]* value for feat,value in features_dictionary.items())


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #We use the formula given in the Project description to update the individual weights.
        features_dictionary = self.featExtractor.getFeatures(state, action)
        #We calculate the difference using the formula given in the project description
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state,action)
        #Then we update each element of the weight dictionary using the formula mentioned in the project description and metrics already calculated above.
        for feat, value in features_dictionary.items():
            self.weights[feat] = self.weights[feat] + self.alpha*difference*value


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
