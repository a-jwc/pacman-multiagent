# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util
import time

from game import Agent

citations = [
    "https://web.stanford.edu/class/archive/cs/cs221/cs221.1196/assignments/pacman/index.html",
    "https://www.youtube.com/watch?v=l-hh51ncgDI",
]

print("Citations: ", citations)
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        distToFood = 0
        distToGhost = 0
        newScore = successorGameState.getScore()
        
        # * sum reciprocal of dist to food
        for food in newFood.asList():
            if manhattanDistance(newPos, food) != 0:
                distToFood += 1/manhattanDistance(newPos, food)

        # * sum reciprocal of dist to ghost
        for state in newGhostStates:
            if manhattanDistance(newPos, state.getPosition()) != 0:
                distToGhost += 1/manhattanDistance(newPos, state.getPosition())

        # * Redundant check
        if distToFood != 0:
            newScore += distToFood
        if distToGhost != 0:
            newScore -= distToGhost

        # * newscore = w1 * distToFood + w2 * distToGhost + w3 * getScore
        return newScore


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.agent = self.index
        self.depthCount = 0
        self.newAction = Directions.STOP
        self.numAgents = gameState.getNumAgents()
        self.legalMoves = gameState.getLegalActions(self.index)
        values = []

        # * minimax driver
        def minimax(state, agent, depth):
            if state.isWin() or state.isLose() or self.depth == depth:
                return self.evaluationFunction(state)
            elif agent == 0:
                return maxValue(state, agent, depth)
            else:
                return minValue(state, agent, depth)

        # * calculate some value for pacman agent
        def maxValue(state, agent, depth):
            maxVal = float('-inf')
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                tempVal = minimax(successor, agent + 1, depth)
                if tempVal > maxVal:
                    maxVal = tempVal
            return maxVal

        # * calculate some value for ghost agents
        def minValue(state, agent, depth):
            # * if next agent is pacman, reset depth and agent
            # * else, agent to next ghost
            if agent + 1 == self.numAgents:
                tempAgent = 0
                depth += 1
            else:
                tempAgent = agent + 1
            minVal = float('inf')
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                minVal = min(minVal, minimax(successor, tempAgent, depth))
            return minVal
        
        # * set self.agent to refer to ghost
        self.agent += 1
        
        for action in self.legalMoves:
            # * next pacman state
            nextState = gameState.generateSuccessor(self.index, action)   
            # * ghost state
            currValue = minimax(nextState, self.agent, self.depthCount)
            values.append(currValue)
        
        # * find max in value
        actionIndex = values.index(max(values))
        return self.legalMoves[actionIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.agent = self.index
        self.depthCount = 0
        self.newAction = Directions.STOP
        self.numAgents = gameState.getNumAgents()
        self.legalMoves = gameState.getLegalActions(self.index)
        values = []

        # * expectimax driver
        def expectimax(state, agent, depth):
            if state.isWin() or state.isLose() or self.depth == depth:
                # print("self.depth", self.depth, "depth", depth)   
                return self.evaluationFunction(state)
            elif agent == 0:
                return maxValue(state, agent, depth)
            else:
                return expValue(state, agent, depth)
            
        # * calculate some value for pacman agent
        def maxValue(state, agent, depth):
            maxVal = float('-inf')
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                tempVal = expectimax(successor, agent + 1, depth)
                if tempVal > maxVal:
                    maxVal = tempVal
            return maxVal

        # * calculate some value for ghost agents
        def expValue(state, agent, depth):
            expVal = 0
            if agent + 1 == self.numAgents:
                tempAgent = 0
                depth += 1
            else:
                tempAgent = agent + 1
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                p = 1/len(state.getLegalActions(agent))
                expVal += p * expectimax(successor, tempAgent, depth)
            return expVal
        
        # * set self.agent to refer to ghost        
        self.agent += 1
        

        for action in self.legalMoves:
            # * next pacman state
            nextState = gameState.generateSuccessor(self.index, action)   
            # * ghost state
            currValue = expectimax(nextState, self.agent, self.depthCount)
            values.append(currValue)
            
        # * find max in value            
        actionIndex = values.index(max(values))
        return self.legalMoves[actionIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    legalMoves = currentGameState.getLegalActions(0)
    scoresList = []
    
    # * Run reflex agent evaluation function for each legal pacman move
    for move in legalMoves:
        successorGameState = currentGameState.generateSuccessor(0, move)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        distToFood = 0
        distToGhost = 0
        newScore = successorGameState.getScore()
        
        # * sum reciprocal of dist to food
        for food in newFood.asList():
            if manhattanDistance(newPos, food) != 0:
                distToFood += 1/manhattanDistance(newPos, food)
                
        # * sum reciprocal of dist to ghost
        for state in newGhostStates:
            if manhattanDistance(newPos, state.getPosition()) != 0:
                distToGhost += 1/manhattanDistance(newPos, state.getPosition())
                
        # * Redundant check
        if distToFood != 0:
            newScore += distToFood
        if distToGhost != 0:
            newScore -= distToGhost

        scoresList.append(newScore)
    
    # * if the scoresList is empty, return the current game state score
    if len(scoresList) != 0:
        newScore = max(scoresList)
    else: 
        newScore = currentGameState.getScore()
    # * newscore = w1 * distToFood + w2 * distToGhost + w3 * getScore
    return newScore


# Abbreviation
better = betterEvaluationFunction
