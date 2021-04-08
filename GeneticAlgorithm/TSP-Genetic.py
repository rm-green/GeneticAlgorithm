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

#TODO: Add timers and graphs
#TODO: See how we can improve performance (8-9 seconds for graphs of size 10-30)

def getInitialPop(G, popSize):
	#Create a random population (list) of individuals (routes)

	pop = []
	for i in range(popSize):
		pop.append(np.random.permutation(G.vertices))
	
	return pop


def determineFitnessAndRank(pop):

	#Creates dictionary of {Route Index: Fitness Score} 
	#Fitness score = reciprocal of route length
	fitnessResults = {}
	for i in range(0, len(pop)):
		fitnessResults[i] = (1 / TSP.cost(G, pop[i]))

	#Sorts the route indicies by fitness score
	popRanked = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

	return popRanked

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

	#Tournament selections
	for i in range(0, len(popRanked) - eliteSize):
		smaller, larger = sorted(random.sample(popRanked, 2))
		selectionResults.append(larger[0])

	
	#ALTERNATIVE RANDOM SELECTION METHOD
	# for i in range(0, len(popRanked) - eliteSize):
	# 	pick = random.randint(0, len(popRanked))
	# 	selectionResults.append(popRanked[i][0])
	return selectionResults #An array of which indexes (from the original unsorted population) will be selected for mating

def getMatingPool(pop, selectionResults):
	#Fill our mating pool
	matingPool = []
	for i in range(0, len(selectionResults)):
		ind = selectionResults[i]
		matingPool.append(pop[ind])

	return matingPool

def breed(parent1, parent2):
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
		if random.random() > crossoverRate:
			child1, child2 = breed(matingPool[i][1:-1], matingPool[i+1][1:-1])
			children.append([0] + child1 + [0])
			children.append([0] + child2 + [0])
		else:
			children.append(matingPool[i])
			children.append(matingPool[i+1])
	
	return children

def mutate(route):
	#Use mutation - Each route has a chance to mutate (random swap of two routes) determined by mutationRate
	#Potentially implement 2-opt or 3-opt here to see if larger mutations affect the algorithm
	i, j = sorted(random.sample(range(1, len(route)-1), 2))
	route[i:j] = route[j-1:i-1:-1]
	
def mutatePop(population, mutationRate, eliteSize):
	#Iterates over an entire population and applies mutation
	mutatedPop = []

	for i in range(0, eliteSize):
		mutatedPop.append(population[i])
	
	for i in range(0, len(population) - eliteSize):
		if random.random() > mutationRate:
			mutate(population[i]) #Mutate in place
		mutatedPop.append(population[i])
	
	return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate, crossoverRate):
	'''
	1. Determine fitness (route lengths)
	2. Select mating pool (choose subset of routes)
	3. Breed (merge pairs of routes to create new routes)
	4. Repeat
	'''
	rankedPop = determineFitnessAndRank(currentGen)
	selectionResults = selection(rankedPop, eliteSize)
	matingPool = getMatingPool(currentGen, selectionResults)
	children = breedPop(matingPool, eliteSize, crossoverRate) #Bottleneck occuring here
	nextGen = mutatePop(children, mutationRate, eliteSize)

	return nextGen

def GeneticAlgorithm(G, popSize, eliteSize, mutationRate, crossoverRate, generations):
	#Get initial population and best distance
	pop = getInitialPop(G, popSize)
	initialDistance = 1 / determineFitnessAndRank(pop)[0][1]

	#Apply the algorithm
	for i in range(0, generations):
		pop = nextGeneration(pop, eliteSize, mutationRate, crossoverRate)
		print(f'Gen {i} Complete')
	
	#Get our new best distance and route
	bestDistance = 1 / determineFitnessAndRank(pop)[0][1]
	print(f"Initial Distance: {initialDistance} \nFinal Distance: {bestDistance}")
	improvementPerc = (1 - (bestDistance/initialDistance))*100
	bestRouteIndex = determineFitnessAndRank(pop)[0][0]
	bestRoute = pop[bestRouteIndex]

	return (improvementPerc, bestRoute)

#Initalise our constants
MIN_SIZE = 100
MAX_SIZE = 1000
INCREMENT = 200
POP_SIZE = 100
ELITE_SIZE = 20
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
GENERATIONS = 200

FILENAME = f'GenAlg_pop={POP_SIZE}_elite={ELITE_SIZE}_mut={MUTATION_RATE}_gens={GENERATIONS}'

#Create the graph, run the algorithm
data = pd.DataFrame(columns=['j','time', 'length', 'improvement percentage'])
i = 0
for j in range(MIN_SIZE, MAX_SIZE, INCREMENT):
	#For profiling purposes
	profile = cProfile.Profile()
	G = TSP.Graph(j, 'asymmetric')
	profile.enable()
	t0 = perf_counter()
	improvementPerc, route = GeneticAlgorithm(G, POP_SIZE, ELITE_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS)
	t1 = perf_counter()
	profile.disable()
	t = t1-t0
	data.loc[i] = [j, t, TSP.cost(G, route), improvementPerc]
	i += 1

	#Output profiling info to a text file
	with open(f'./GeneticAlgorithm/GeneticProfilings/{FILENAME}_Profiling.txt', 'a+') as stream:
		ps = pstats.Stats(profile, stream=stream)
		ps.sort_stats('cumtime')
		ps.print_stats(20)
	
#Output dataframe to another file
data.to_csv(f'./GeneticAlgorithm/GeneticOutputs/{FILENAME}_Output.csv', mode = 'w')

