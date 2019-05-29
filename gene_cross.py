from numpy import random


MAX_HIDDEN_LAYERS = 20
MAX_NEURONS_PER_LAYER = 25
MAX_NUMBER_OF_ITERATIONS = 20000
MUTATION_CHANCE = 100#Higher number, lower chance, 100 = 1%

activationFunctionList = ["identity","logistic","tanh","relu"]
solverList = ["lbfgs","sgd","adam"]

def limitAlpha(alpha):
    if alpha > 1.0:
        alpha=1.0
    elif alpha <= 0.0:
        alpha = 0.0
    return alpha

class Gene:
    hiddenLayerSizes = []
    nHiddenLayers = len(hiddenLayerSizes)
    activationFunction = ""
    solver = ""
    alpha = 0.0
    n_iter = 0
    
    fitness = 0.0
    f1 = 0.0
    auc = 0.0
    accuracy = 0.0
    
    def __init__(self, hiddenLayerSizes, activationFunction, solver, alpha, n_iter):
        self.hiddenLayerSizes = hiddenLayerSizes
        self.nHiddenLayers = len(self.hiddenLayerSizes)
        self.activationFunction = activationFunction
        self.solver = solver
        self.alpha = alpha
        self.n_iter = n_iter
      
    def printAll(self):
        print("Number of Hidden Layers: ", self.nHiddenLayers)
        print("hiddenLayerSizes = ", self.hiddenLayerSizes)
        print("activation_ = \"", self.activationFunction, "\"")
        #print(self.solver)
        print("F1:", self.fitness)
        print("AUC:", self.auc)
        print("ACCURACY:", self.accuracy)

    def fprintAll(self, f):
        f.write("n_layers = " + str(self.nHiddenLayers) + "\n")
        f.write("\n\n")
        f.write("hiddenLayerSizes = " + str(self.hiddenLayerSizes) + "\n")
        f.write("activation_ = \"" + str(self.activationFunction) + "\"" + "\n")
        #print(self.solver)
        f.write("alpha_ = "  + str(self.alpha) + "\n")
        f.write("n_iter = " + str(self.n_iter) + "\n")
        f.write("\n\n")
        f.write("F1 = " + str(self.fitness) + "\n")
        f.write("AUC = "  + str(self.auc) + "\n")
        f.write("ACCURACY = " + str(self.accuracy) + "\n")

    def setRandom(self):
    
        #modify length
        oldLen = len(self.hiddenLayerSizes)
        self.nHiddenLayers = random.randint(1,MAX_HIDDEN_LAYERS)
        newLen = self.nHiddenLayers
        if newLen > oldLen:
            for i in range(newLen - oldLen):
                self.hiddenLayerSizes.append(
                    random.randint(1,MAX_NEURONS_PER_LAYER))
        elif oldLen > newLen:
            for i in range(oldLen-newLen):
                self.hiddenLayerSizes.pop()
    
        self.activationFunction = activationFunctionList[random.randint(4)]
    
        self.solver = solverList[random.randint(3)]
        self.alpha = random.uniform(0, 1)
    
        self.n_iter = random.randint(MAX_NUMBER_OF_ITERATIONS)
    
        #modify hiddenLayerSizes
        for i in range(len(self.hiddenLayerSizes)):
            self.hiddenLayerSizes[i] = random.randint(1, MAX_NEURONS_PER_LAYER)

def crossLayers(list1, list2, length = 0):
    len_list1 = len(list1)
    len_list2 = len(list2)
    if length == 0:
        length = max(len_list1, len_list2)
    
    #Zero padding
    if len_list1 > len_list2:
        for i in range(len_list1-len_list2):
            list2.append(1)
    elif len_list2 > len_list1:
        for i in range(len_list2-len_list1):
            list1.append(1)
    
    #average recombination
    ret_list = []
    
    for i in range(length):
        if list1[i] != 1 and list2[i] != 1:
            x = int((list1[i]+list2[i])/2)
            if x == 0:
                x=1
            ret_list.append(x)

        elif list1[i] == 1:
            ret_list.append(list2[i])
        elif list2[i] == 1:
            ret_list.append(list1[i])
        elif list1[i] == 1  and list2[i] == 1:
            ret_list.append(1)



    return ret_list

def mutator(gene):
    choice = random.randint(5)
    if choice == 0:
        #modify length
        oldLen = len(gene.hiddenLayerSizes)
        gene.nHiddenLayers = random.randint(1, MAX_HIDDEN_LAYERS)
        newLen = gene.nHiddenLayers
        if newLen > oldLen:
            for i in range(newLen - oldLen):
                gene.hiddenLayerSizes.append(random.randint(1, MAX_NEURONS_PER_LAYER))
        elif oldLen > newLen:
            for i in range(oldLen-newLen):
                gene.hiddenLayerSizes.pop()
    if choice == 1:
        gene.activationFunction=activationFunctionList[random.randint(4)]
    if choice == 2:
        gene.alpha = random.uniform(0,1)
    if choice == 3:
        gene.n_iter = random.randint(100,MAX_NUMBER_OF_ITERATIONS)
    if choice == 4:
        #modify hiddenLayerSizes
        for i in range(len(gene.hiddenLayerSizes)):
            gene.hiddenLayerSizes[i] = random.randint(1,MAX_NEURONS_PER_LAYER)


def crossGenes(gene1, gene2):
    #print("\n\nCROSSING STARTED\n\n")
    #gene1.printAll()
    #gene2.printAll()
    #print("\n\nVALUES PRINTED\n\n")
    new_hiddenLayerSizes = crossLayers(gene1.hiddenLayerSizes, gene2.hiddenLayerSizes)
    new_nHiddenLayers = len(new_hiddenLayerSizes)
    new_activationFunction = random.choice([gene1.activationFunction, gene2.activationFunction], 1, p=[0.5,0.5])
    #print(new_activationFunction)
    new_solver = "adam"
    #new_alpha = (gene1.alpha+gene2.alpha)/2.0
    #new_n_iter = int((gene1.n_iter+gene2.n_iter)/2)
    new_alpha = random.choice([gene1.alpha, gene2.alpha])
    new_n_iter = random.choice([gene1.n_iter, gene2.n_iter])
    ret_gene = Gene(new_hiddenLayerSizes, new_activationFunction[0], new_solver, new_alpha, new_n_iter)

    if random.randint(MUTATION_CHANCE) == 0:
        #print("MUTATION")
        mutator(ret_gene)
    return ret_gene


#turns the number of errors above threshold into 1 and below into 0
def binarize(y, threshold=1):
    l_y = len(y)
    for i in range(l_y):
        #print("\033[K", end='')
        print("Binarizing:", int((i/l_y)*100), "%", end="\r")
        #print("Before: ", y[i])
        if y[i] > threshold:
            y[i] = 1
        else:
            y[i] = 0
        #print("After: ", y[i])
    print("Binarizing done.")

"""
gene1 = Gene([10,10,20,34],activationFunctionList[0], solverList[0], 0.4, 2000)
gene2 = Gene([10,20,20],activationFunctionList[2], solverList[1], 0.5, 1500)

gene_new = crossGenes(gene1,gene2)

gene_new.printAll()

gene3 = Gene([0],'.','.',0.0,0)
gene3.setRandom()
print("gene3:")
gene3.printAll()
"""
