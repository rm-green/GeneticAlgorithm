import TSP
import pandas as pd
import numpy as np
import operator
import random
from time import perf_counter

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

	#Randomly choose the remaining routes
	for i in range(0, len(popRanked) - eliteSize):
		pick = 100*np.random.random()
		for i in range(0, len(popRanked)):
			if pick <= df.iat[i,3]:
				selectionResults.append(popRanked[i][0])
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

	#Routes are constrained in that they must visit all routes exactly once
	#Therefore we use "ordered crossover" to breed our routes
	child = []
	childParent1 = []
	childParent2 = []

	#Choose a random subset of the first route
	gene1 = int(random.random() * len(parent1))
	gene2 = int(random.random() * len(parent1))
	start = min(gene1, gene2)
	end = max(gene1, gene2)

	#Take that subset and append it to a new array
	for i in range(start, end):
		childParent1.append(parent1[i])
	
	#Take the order of the remaining vertices from the second route and append it to a separate array
	childParent2 = [i for i in parent2 if i not in childParent1]

	#Combine the two partial arrays to create a child
	child = childParent1 + childParent2

	return child

def breedPop(matingPool, eliteSize):
	#Creates a new population from the mating pool
	children = []
	length = len(matingPool) - eliteSize

	#Make a separate array containing the mating pool in random order
	pool = random.sample(matingPool, len(matingPool))

	#Automatically bring the elites into the next generation
	#This ensures that future generations can only improve/never regress
	for i in range(0, eliteSize):
		children.append(matingPool[i])
	
	#Breed the mating pool (moving inwards from the start and end) and add to the next generation
	for i in range(0, length):
		child = breed(pool[i], pool[len(matingPool) - i - 1])
		children.append(child)
	
	return children

def mutate(route, mutationRate):
	#Use mutation - Each route has a chance to mutate (random swap of two routes) determined by mutationRate
	#Potentially implement 2-opt or 3-opt here to see if larger mutations affect the algorithm
	for swapped in range(len(route)):
		if(random.random() < mutationRate):
			swapWith = int(random.random() * len(route))

			v1 = route[swapped]
			v2 = route[swapWith]

			route[swapped] = v2
			route[swapWith] = v1
	
	return route

def mutatePop(population, mutationRate):
	#Iterates over an entire population and applies mutation
	mutatedPop = []

	for i in range(0, len(population)):
		mutatedRoute = mutate(population[i], mutationRate)
		mutatedPop.append(mutatedRoute)
	
	return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
	'''
	1. Determine fitness (route lengths)
	2. Select mating pool (choose subset of routes)
	3. Breed (merge pairs of routes to create new routes)
	4. Repeat
	'''
	rankedPop = determineFitnessAndRank(currentGen)
	selectionResults = selection(rankedPop, eliteSize)
	matingPool = getMatingPool(currentGen, selectionResults)
	children = breedPop(matingPool, eliteSize)
	nextGen = mutatePop(children, mutationRate)

	return nextGen

def GeneticAlgorithm(G, popSize, eliteSize, mutationRate, generations):
	#Get initial population and best distance
	pop = getInitialPop(G, popSize)
	initialDistance = 1 / determineFitnessAndRank(pop)[0][1]

	#Apply the algorithm
	for i in range(0, generations):
		pop = nextGeneration(pop, eliteSize, mutationRate)
		print(f'Gen {i} Complete')
	
	#Get our new best distance and route
	print(f"Initial Distance: {initialDistance} \nFinal Distance: {1 / determineFitnessAndRank(pop)[0][1]}")
	bestRouteIndex = determineFitnessAndRank(pop)[0][0]
	bestRoute = pop[bestRouteIndex]

	return (initialDistance, bestRoute)

#Create the graph, run the algorithm
data = pd.DataFrame(columns=['j','time', 'initial length', 'length'])
i = 0
for j in range(10, 31, 10):
	G = TSP.Graph(j, 'asymmetric')
	t0 = perf_counter()
	initialDistance, route = GeneticAlgorithm(G, 100, 5, 0.01, 50)
	t1 = perf_counter()
	t = t1-t0
	data.loc[i] = [j, t, initialDistance, TSP.cost(G, route)]
	i += 1

print(data)
