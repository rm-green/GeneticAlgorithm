import TSP
import pandas as pd
import numpy as np
import operator
import random
from time import perf_counter
import cProfile
import pstats
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

#TODO: Add timers and graphs
#TODO: See how we can improve performance (8-9 seconds for graphs of size 10-30)

def getInitialPop(G, popSize):
	#Create a random population (list) of individuals (routes)

	pop = []

	#We need routes that start and end at 0 so we fix them to the start and end
	verts = G.vertices.copy()
	verts.remove(0)
	for i in range(popSize):
		pop.append([0] + list(np.random.permutation(verts)) + [0])
	
	return pop


def determineFitnessAndRank(G, pop):

	#Creates dictionary of {Route Index: Fitness Score} 
	#Fitness score = reciprocal of route length
	fitnessResults = {}
	for i in range(0, len(pop)):
		fitnessResults[i] = (1 / TSP.cost(G, pop[i]))

	#Sorts the route indicies by fitness score
	popRanked = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

	return popRanked #array of tuples (index, score)

def selection(popRanked, eliteSize):

	#Start selecting which individuals (routes) will breed
	#Use Elitisim (The fittest routes will automatically be added to the breeding pool)
	selectionResults = []

	#Put index/fitness dict into dataframe
	df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])

	#Record the cumulative fitness score as you move down the table
	df['cum_sum'] = df.Fitness.cumsum()

	#Also record this as a cumulative percentage
	df['cum_perc'] = 100*(df.Fitness.cumsum()/df.Fitness.sum())

	for i in range(0, eliteSize):
		#Add our n number of elite routes
		selectionResults.append(popRanked[i][0])
	
	#Weighted roulette selections
	max = df['cum_sum'].iloc[-1]
	for i in range(len(popRanked) - eliteSize):
		pick = random.uniform(0, max)
		current = max
		for chromosome in popRanked:
			current -= chromosome[1]
			if current < pick:
				selectionResults.append(chromosome[0])
				break
	
	return selectionResults #An array of which indexes (from the original unsorted population) will be selected for mating

def getMatingPool(pop, selectionResults):
	#Fill our mating pool
	matingPool = []
	for i in range(0, len(selectionResults)):
		ind = selectionResults[i]
		matingPool.append(pop[ind])

	return matingPool

def breed(parent1, parent2):
	#Cycle Crossover method
	#Uses a lookup table to reduce runtime (list.locate() is O(n), lookup is (O(1)))
	#Preserves the start and end nodes (Good choice for our constraints)
	lookup1 = lookup2 = {}
	for i in range(len(parent1)):
		lookup1[parent1[i]] = i
	for i in range(len(parent2)):
		lookup2[parent2[i]] = i

	cycles = [-1 for i in range(len(parent1))]
	cycleNumber = 1
	cycleStart = (i for index, value in enumerate(cycles) if value < 0)

	for position in cycleStart:
		while cycles[position] < 0:
			cycles[position] = cycleNumber
			position = lookup1[parent2[position]]
		cycleNumber += 1
	
	child1 = [parent1[i] if n%2 else parent2[i] for i, n in enumerate(cycles)]
	child2 = [parent2[i] if n%2 else parent1[i] for i, n in enumerate(cycles)]

	return child1, child2

def breedPop(matingPool, eliteSize, crossoverRate):
	#Creates a new population from the mating pool
	children = []
	length = len(matingPool) - eliteSize

	#Automatically bring the elites into the next generation
	#This ensures that future generations can only improve/never regress
	for i in range(0, eliteSize):
		children.append(matingPool[i])
	
	#Breed the mating pool and add to the next generation
	for i in range(0, length-1, 2):
		if random.random() < crossoverRate:
			child1, child2 = breed(matingPool[i], matingPool[i+1])
			children.append(child1)
			children.append(child2)
		else:
			children.append(matingPool[i])
			children.append(matingPool[i+1])
	
	return children

def mutate(route):
	#Reverses a random length section of the route
	#The start and end nodes cannot be selected as part of this section
	i, j = sorted(random.sample(range(1, len(route)-1), 2))
	route[i:j] = route[j-1:i-1:-1]
	
def mutatePop(population, mutationRate, eliteSize):
	#Iterates over an entire population and applies mutation
	mutatedPop = []

	for i in range(0, eliteSize):
		mutatedPop.append(population[i])
	
	for i in range(eliteSize, len(population)-1):
		if random.random() < mutationRate:
			mutate(population[i]) #Mutate in place
		mutatedPop.append(population[i])
	
	return mutatedPop

def nextGeneration(G, currentGen, eliteSize, mutationRate, crossoverRate):
	'''
	1. Determine fitness (route lengths)
	2. Select mating pool (choose subset of routes)
	3. Breed (merge pairs of routes to create new routes)
	4. Repeat
	'''
	rankedPop = determineFitnessAndRank(G, currentGen)
	selectionResults = selection(rankedPop, eliteSize)
	matingPool = getMatingPool(currentGen, selectionResults)
	children = breedPop(matingPool, eliteSize, crossoverRate)
	nextGen = mutatePop(children, mutationRate, eliteSize)

	return nextGen

#------------------------------------------------------#
'''Runs the genetic algorithm over a given number of generations. Returns the best route, 
	best cost, and an array containing best cost of each generation'''

def GeneticAlgorithm(G, popSize = 200, eliteSize = 5, mutationRate = 0.2, crossoverRate = 0.85, generations = 500):
	#Get initial population and best distance
	pop = getInitialPop(G, popSize)
	initialDistance = 1 / determineFitnessAndRank(G, pop)[0][1]
	distanceHistory = {}

	#Apply the algorithm
	for i in range(0, generations):
		pop = nextGeneration(G, pop, eliteSize, mutationRate, crossoverRate)
		print(f'Gen {i} Complete')
		distanceHistory[i] = (1 / determineFitnessAndRank(G, pop)[0][1])
	
	#Get our new best distance and route
	bestDistance = 1 / determineFitnessAndRank(G, pop)[0][1]
	print(f"Initial Distance: {initialDistance} \nFinal Distance: {bestDistance}")
	improvementPerc = (1 - (bestDistance/initialDistance))*100
	bestRouteIndex = determineFitnessAndRank(G, pop)[0][0]
	bestRoute = pop[bestRouteIndex]

	return (bestRoute, bestDistance, distanceHistory)

#------------------------------------------------------#
'''Produces a graph that shows the best route cost against the current generation number of the 
	genetic algorithm alongside the greedy cost (so we can see how long it takes to become 
	better than greedy)'''

def ConvergenceTest(n = 50, graphType = "symmetric", popSize = 200, eliteSize = 10, mutationRate = 0.4, crossoverRate = 0.85, generations = 500):
	myG = TSP.Graph(n, graphType)
	geneticRoute, geneticCost, distanceHistory = GeneticAlgorithm(myG, popSize, eliteSize, mutationRate, crossoverRate, generations)
	greedyRoute, greedyCost = TSP.greedy_nearest_neighbour(myG)
	greedyCosts = [greedyCost for i in range(generations)]
	plt.figure(figsize=(15,5))
	plt.plot(list(distanceHistory.keys()), list(distanceHistory.values()))
	plt.plot(list(distanceHistory.keys()), greedyCosts)
	print(f'Greedy Cost: {greedyCost}')	#Plot the result here

	print(f'Greedy Route: {greedyRoute}')
	print(f'Genetic Cost: {geneticCost}')
	print(f'Genetic Route: {geneticRoute}')
	plt.show()

#------------------------------------------------------#
#------------------------------------------------------#

'''These two functions run the genetic algorithm and greedy algorithm on a graph for a number of 
repetitions and returns the average time taken and route quality in a Panda DataFrame'''

def testSpeedAgainstGreedy(repetitions = 5, n = 50, graphType = "symmetric", popSize = 200, eliteSize = 10, mutationRate = 0.4, crossoverRate = 0.85, generations = 300):
	data = pd.DataFrame(columns=['n', 'population size', 'mutation rate', 'crossover rate', 'generations', 'greedy time', 'genetic time'])
	i = 0
	for _ in range(repetitions):
		myG = TSP.Graph(n, graphType)
		t0 = perf_counter()
		greedyRoute, greedyCost = TSP.greedy_nearest_neighbour(myG)
		t1 = perf_counter()
		greedyTime = t1 - t0
		t0 = perf_counter()
		geneticRoute, geneticCost, distanceHistory = GeneticAlgorithm(myG, popSize, eliteSize, mutationRate, crossoverRate, generations)
		t1 = perf_counter()
		geneticTime = t1-t0
		data.loc[i] = [n, popSize, mutationRate, crossoverRate, generations, greedyTime, geneticTime]
		i += 1
	
	return data
		
def testQualityAgainstGreedy(repetitions = 5, n = 50, graphType = "symmetric", popSize = 200, eliteSize = 10, mutationRate = 0.4, crossoverRate = 0.85, generations = 300):
	data = pd.DataFrame(columns=['n', 'population size', 'mutation rate', 'crossover rate', 'generations', 'greedy cost', 'genetic cost', 'quality'])
	i = 0
	
	for _ in range(repetitions):
		myG = TSP.Graph(n, graphType)
		geneticRoute, geneticCost, distanceHistory = GeneticAlgorithm(myG, popSize, eliteSize, mutationRate, crossoverRate, generations)
		greedyRoute, greedyCost = TSP.greedy_nearest_neighbour(myG)
		data.loc[i] = [n, popSize, mutationRate, crossoverRate, generations, greedyCost, geneticCost, 1 / (geneticCost/greedyCost)]
		i += 1
	
	return data

#------------------------------------------------------#
#------------------------------------------------------#

'''Test functions: Use these or just copy/paste the body into the notebook 
and plug in n, min, max, and increment. Plotting could be improved'''

def testSpeedVaryingGraphSizes(min = 10, max = 301, increment = 50, graphType = "symmetric"):
	frames = [testSpeedAgainstGreedy(graphType = graphType, n = n) for n in range(min, max, increment)]
	concatenatedFrames = pd.concat(frames)
	result = concatenatedFrames.groupby('n').agg('mean', 'std')
	plt.plot(result.index, result['genetic time'])
	plt.plot(result.index, result['greedy time'])
	plt.show()

def testSpeedVaryingPopulationSizes(min=100, max=501, increment=100, graphType = "symmetric"):
	frames = [testSpeedAgainstGreedy(graphType = graphType, popSize = n) for n in range(min, max, increment)]
	concatenatedFrames = pd.concat(frames)
	result = concatenatedFrames.groupby('population size').agg('mean', 'std')
	plt.plot(result.index, result['genetic time'])
	plt.plot(result.index, result['greedy time'])
	plt.show()

#Added addtional test functions
def testSpeedVaryingCrossoverRates(min=0.5, max=5, increment=0.1, graphType = "symmetric"):
	frames = [testSpeedAgainstGreedy(graphType = graphType, crossoverRate = n) for n in range(min, max, increment)]
	concatenatedFrames = pd.concat(frames)
	result = concatenatedFrames.groupby('crossover rates').agg('mean', 'std')
	plt.plot(result.index, result['genetic time'])
	plt.plot(result.index, result['greedy time'])
	plt.show()
	
def testSpeedVaryingMutationRates(min=0.2, max=2, increment=0.1, graphType = "symmetric"):
	frames = [testSpeedAgainstGreedy(graphType = graphType, mutationRate = n) for n in range(min, max, increment)]
	concatenatedFrames = pd.concat(frames)
	result = concatenatedFrames.groupby('mutation rates').agg('mean', 'std')
	plt.plot(result.index, result['genetic time'])
	plt.plot(result.index, result['greedy time'])
	plt.show()
	
def testSpeedVaryingGenerations(min=100, max=501, increment=50, graphType = "symmetric"):
	frames = [testSpeedAgainstGreedy(graphType = graphType, generations = n) for n in range(min, max, increment)]
	concatenatedFrames = pd.concat(frames)
	result = concatenatedFrames.groupby('varying generations').agg('mean', 'std')
	plt.plot(result.index, result['genetic time'])
	plt.plot(result.index, result['greedy time'])
	plt.show()

def testQualityVaryingGraphSizes(min = 10, max = 101, increment = 10, graphType = "symmetric"):
	frames = [testQualityAgainstGreedy(graphType = graphType, n = n) for n in range(min, max, increment)]
	concatenatedFrames = pd.concat(frames)
	result = concatenatedFrames.groupby('n').agg('mean', 'std')
	plt.plot(result.index, result['quality'])
	plt.show()

testQualityVaryingGraphSizes(10, 21, 10)
#TODO: Implement other quality tests
#TODO: Move tests to separate file
