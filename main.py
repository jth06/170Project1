import numpy as ny
import math
from queue import PriorityQueue
import copy

class Tree:
    
    def __init__(self, initial_state, goal_state, operators, Algor):    #all variables passed in are necessary
        self.initial_state = initial_state  # so the code knows where to start
        self.goal_state = goal_state        # so the code knows what state it's looking for and knows when to cease searching
        self.operators = operators          # directions the blank space can move
        self.Algor = Algor                  # used to identify which algorithm is being used
        self.explored = set()               # hash table used to store what nodes have been explored by the algorithm already
        self.frontier = PriorityQueue()     # priority queue organized by cost, so the algorithm which node to explore next
        self.frontierCheck = set()          # used to prevent repeated states in the frontier priority queue
        self.maxQueueSize = 0               # keeps track of the maximum queue size at any point in the search
        self.NodesExpanded = 0              # keeps track of how many nodes are expanded in the algorithm

    def validOps(self, currNode): # checks to see what movements are possible
        currPos = ny.where(currNode.current_state == 0)

        possOps = ny.empty([0,0])

        if (currPos[0]/3 >= 1): # checks if blank space can move up
            possOps = ny.append(possOps, [1])
        
        if (currPos[0]%3 < 2): # checks if blank space can move right
            possOps = ny.append(possOps, [2])

        if (currPos[0]%3 > 0): # checks if blank space can move left
            possOps = ny.append(possOps, [3])
        
        if (currPos[0]/3 < 2 and currPos[0]/3 >= 0): # checks if blank space can move down
            possOps = ny.append(possOps, [4])
        
        return possOps

    def addToExplored(self, exploredState): # adds node to explored state
        self.explored.add(exploredState)
        self.NodesExpanded += 1

    def addToFrontier(self, frontierNode):  # adds node to frontier
        self.frontier.put((frontierNode))
        if (self.frontier.qsize() > self.maxQueueSize):
            self.maxQueueSize = self.frontier.qsize()
        self.frontierCheck.add(tuple(frontierNode.current_state))

    def validTransition(self, currentNode): # checks to see what nodes have not been explored yet of the possible transitions from validOps
        possOps = self.validOps(currentNode)

        validOps = ny.empty([0,0])

        #checking for up movement possibility
        currIndex = ny.where(possOps == 1)

        currBlank = ny.where(currentNode.current_state == 0)
        blankIndex = currBlank[0]
        swappedIndex = 0

        if (currIndex[0] == 0):

            swappedIndex = blankIndex - 3
            swappedArray = currentNode.current_state

            temp = swappedArray[swappedIndex]
            swappedArray[swappedIndex] = 0
            swappedArray[blankIndex] = temp
            convertedArray = tuple(swappedArray)

            if convertedArray in self.explored:
                {}

            else:
                validOps = ny.append(validOps, 1)
        
        #checking for right movement possibility
        currIndex = ny.where(possOps == 2)

        if (currIndex[0] == 0 or currIndex[0] == 1):

            swappedIndex = blankIndex + 1
            swappedArray = currentNode.current_state

            temp = swappedArray[swappedIndex]
            swappedArray[swappedIndex] = 0
            swappedArray[blankIndex] = temp
            convertedArray = tuple(swappedArray)

            if convertedArray in self.explored:
                {}

            else:
                validOps = ny.append(validOps, 2)

        #checking for left movement possibility
        currIndex = ny.where(possOps == 3)

        if (currIndex[0] == 0 or currIndex[0] == 1 or currIndex[0] == 2):

            swappedIndex = blankIndex - 1
            swappedArray = currentNode.current_state

            temp = swappedArray[swappedIndex]
            swappedArray[swappedIndex] = 0
            swappedArray[blankIndex] = temp
            convertedArray = tuple(swappedArray)

            if convertedArray in self.explored:
                {}

            else:
                validOps = ny.append(validOps, 3)

        #checking for down movement possibility
        currIndex = ny.where(possOps == 4)

        if (currIndex[0] == 0 or currIndex[0] == 1 or currIndex[0] == 2 or currIndex[0] == 3 ):

            swappedIndex = blankIndex + 3
            swappedArray = currentNode.current_state

            temp = swappedArray[swappedIndex]
            swappedArray[swappedIndex] = 0
            swappedArray[blankIndex] = temp
            
            convertedArray = tuple(swappedArray)

            if convertedArray in self.explored:
                {}

            else:
                validOps = ny.append(validOps, 4)

        return validOps

    def MisplacedCost(self, current_state): # computes h(n) for Misplaced Tile Heuristic
        numMisplaced = 0

        currIndex = ny.where(current_state == 0)
        
        for x in range(8):
            currIndex = ny.where(current_state == x+1)

            if (currIndex[0] != x):
                numMisplaced += 1
        
        return numMisplaced

    def EuclideanCost(self, current_state): # computes h(n) for Euclidean Distance Heuristic
        EuclideanCost = 0

        currIndex = 0

        for x in range(8):
            currIndex = ny.where(current_state == x+1)
            goalIndex = ny.where(self.goal_state == x+1)

            xvalCurr = currIndex[0]%3
            xvalGoal = goalIndex[0]%3

            yvalCurr = currIndex[0]/3
            yvalGoal = goalIndex[0]/3

            EuclideanCost += math.sqrt((xvalCurr-xvalGoal)**2 + (yvalCurr-yvalGoal)**2)

        return EuclideanCost
    
    def UniformCS(self, current_Node):  #Passes in root node and starts the Uniform Cost Search Algorithm, checking to see if heuristics need to be computed as well
        while True:
            if(self.frontier.empty()):
                return -1
            USTNode = self.frontier.get()

            if (ny.array_equal(USTNode.current_state, self.goal_state)):
                
                print("Tracing from the goal state to the initial state:")
                print()

                self.printTrace(USTNode)
                
                return USTNode
            

            USTNodeTup = tuple(USTNode.current_state)

            self.addToExplored(USTNodeTup)

            copyUSTNode = copy.deepcopy(USTNode)

            possibleTransitions = self.validTransition(copyUSTNode)

            for x in possibleTransitions:
                currCopy = copy.deepcopy(USTNode.current_state)
                childState = self.movedState(currCopy, x)

                if(self.Algor == 1):
                    childNode = Node(childState, USTNode, USTNode.gn+1)
                    childNode.gn = USTNode.gn+1

                elif(self.Algor == 2):
                    MisplacedC = self.MisplacedCost(childState)
                    childNode = Node(childState, USTNode, MisplacedC+USTNode.gn+1)
                    childNode.gn = USTNode.gn+1
                    childNode.hn = MisplacedC
                
                elif(self.Algor == 3):
                    EuclideanC = self.EuclideanCost(childState)
                    childNode = Node(childState, USTNode, EuclideanC+USTNode.gn+1)
                    childNode.gn = USTNode.gn+1
                    childNode.hn = EuclideanC

                # checks to see if repeated state is already in frontier
                if (tuple(childState) in self.frontierCheck):
                    {}
                else:
                    # pushes childNode to frontier
                    self.addToFrontier(childNode)
                # if tuple(self.goal_state) in self.explored:



    def movedState(self, current_state, operation): # computes the state that would be created if the blank space were to move in the passed in direction
        newState = current_state

        currIndex = ny.where(current_state == 0)
        swapIndex = 0

        if (operation == 1):
            swapIndex = currIndex[0] - 3
            temp = newState[swapIndex]
            
            newState[swapIndex] = 0
            newState[currIndex] = temp

        elif (operation == 2):
            swapIndex = currIndex[0] + 1
            temp = newState[swapIndex]
            
            newState[swapIndex] = 0
            newState[currIndex] = temp

        elif (operation == 3):
            swapIndex = currIndex[0] - 1
            temp = newState[swapIndex]
            
            newState[swapIndex] = 0
            newState[currIndex] = temp

        elif (operation == 4):
            swapIndex = currIndex[0] + 3
            temp = newState[swapIndex]
            
            newState[swapIndex] = 0
            newState[currIndex] = temp

        return newState

    def printExpandedState(self, curr_Node):
        print("The best state to expand with g(n) =", end =" ") 
        print(curr_Node.gn, end =" ")
        print(" and h(n) =", end =" ") 
        print(curr_Node.hn, end =" ")
        print(" is...")

        print(curr_Node.current_state[0], end =" ")
        print(curr_Node.current_state[1], end =" ")
        print(curr_Node.current_state[2])

        print(curr_Node.current_state[3], end =" ")
        print(curr_Node.current_state[4], end =" ")
        print(curr_Node.current_state[5])

        print(curr_Node.current_state[6], end =" ")
        print(curr_Node.current_state[7], end =" ")
        print(curr_Node.current_state[8], end =" ")
        print("   Expanding this node...")
        print()

    def printTrace(self, curr_Node):    # prints trace from initial node to goal node
        while curr_Node != None:
            print("The best state to expand with g(n) =", end =" ") 
            print(curr_Node.gn, end =" ")
            print(" and h(n) =", end =" ") 
            print(curr_Node.hn, end =" ")
            print(" is...")

            print(curr_Node.current_state[0], end =" ")
            print(curr_Node.current_state[1], end =" ")
            print(curr_Node.current_state[2])

            print(curr_Node.current_state[3], end =" ")
            print(curr_Node.current_state[4], end =" ")
            print(curr_Node.current_state[5])

            print(curr_Node.current_state[6], end =" ")
            print(curr_Node.current_state[7], end =" ")
            print(curr_Node.current_state[8], end =" ")
            print("   Expanding this node...")
            print()
            curr_Node = curr_Node.parent_node
        

class Node:     # node class used to keep track of states, cost, and parent node values
    def __init__(self, current_state, parent_node, cost):
        self.current_state = current_state
        self.parent_node = parent_node
        self.cost = cost
        self.hn = 0
        self.gn = 0

    def __lt__(self, other):    # added so that we can compare values for frontier nodes and prioritize ones with a lower cost
        selfPriority = self.cost
        otherPriority = other.cost
        
        if (otherPriority != selfPriority):
            return selfPriority < otherPriority

        elif (otherPriority == selfPriority):
            selfGN = self.gn
            otherGN = other.gn
            return selfGN < otherGN


def main(): # calls above tree functions and builds interface for user
    print("Welcome to 862013504 8 puzzle solver.")
    print("Type \"1\" to use a default puzzle, or \"2\" to enter your own puzzle. ")

    GS = ny.array([1,2,3,4,5,6,7,8,0])

    whichPuz = int(input())

    A = None

    if (whichPuz == 1):
        A = ny.array([8,7,1,6,0,2,5,4,3])
        #print(A)
    
    elif (whichPuz == 2):
        print("Enter your puzzle, use a zero to represent the blank")
        print("Enter the first row, use spaces between numbers")
        
        getRow = ny.array([input().split()],int)

        print("Enter the second row, use spaces between numbers")

        getRow = ny.append(getRow, (ny.array([input().split()],int)))

        print("Enter the first row, use spaces between numbers")

        getRow = ny.append(getRow, (ny.array([input().split()],int)))
        print(getRow)
        
        A = getRow

    print()
    print("Enter your choice of algorithm")
    print("Uniform Cost Search")
    print("A* with the Misplaced Tile heuristic")
    print("A* with the Euclidean distance heuristic")

    whichAlg = int(input())
    print()

    ops = ny.array([1,2,3,4]) # 1 is up, 2 is right, 3 is left, 4 is down

    rootNode = Node(A, None, 0)

    initTree = Tree(rootNode, GS, ops, whichAlg)

    if (whichAlg == 2):
        rootNode.hn = initTree.MisplacedCost(rootNode.current_state)

    elif (whichAlg == 3):
        rootNode.hn = initTree.EuclideanCost(rootNode.current_state)

    initTree.addToFrontier(rootNode)

    initTree.UniformCS(rootNode)

    print("To solve this problem, the search algorithm expanded a total of", end =" ")
    print(initTree.NodesExpanded, end =" ")
    print("nodes.")

    print("The maximum number of nodes in the queue at any one time:", end =" ")
    print(initTree.maxQueueSize, end =".")
    print()
    print()


main()
