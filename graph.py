import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import timeit

class Network(object):
    def __init__(self , num_nodes):
        self.adj = {i:set() for i in range (num_nodes)}

    def add_edge(self , i , j):
        self.adj[i].add(j)
        self.adj[j].add(i)

    def neighbors(self , i):
        return self.adj[i]
    
    def edge_list(self):
        return [(i,j)for i in self.adj for j in self.adj[i] if i < j]
    
    def num_edges(self):
        return sum([len(self.adj[i]) for i in self.adj])//2
    
    def find_component(self, i):
        c = set()
        stack = [i]
        while stack:
            node = stack.pop()
            c.add(node)
            stack.extend(self.neighbors(node) - c)
        return c
    
    def degree_distributions_individual(self):
        return np.array([len(self.adj[i]) for i in self.adj])
    
    def degree_distributions(self):
        degrees = [len(self.adj[i]) for i in self.adj]
        return pd.Series(degrees).value_counts().sort_index()
    
    def friends_degree_distribution(self):
        return np.array([np.mean([len(self.adj[j]) for j in self.adj[i]]) for i in self.adj])
    

class random_graph(Network):
    def __init__(self , num_nodes , p):
        super().__init__(num_nodes)
        for i in range(num_nodes):
            for j in range(i+1 , num_nodes):
                if random.random() > (1-p):
                    self.add_edge(i , j)

class lambda_graph(Network):
    def __init__(self , num_nodes , lambda_):
        super().__init__(num_nodes)
        num_edges = lambda_
        num_added_edges = self.num_edges()
        while num_added_edges < num_edges:
            i = random.randint(0 , num_nodes-1)
            j = random.randint(0 , num_nodes-1)
            if i != j:
                if j not in self.adj[i]:
                    num_added_edges += 1
                self.add_edge(i , j)

class configuration_graph(Network):
    def __init__(self , num_nodes , degree_sequence):
        super().__init__(num_nodes)
        S = np.array([ i for i in range (num_nodes) for _ in range (degree_sequence[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)

class poisson_configuration_graph(Network):
    def __init__(self , num_nodes , lambda_):
        super().__init__(num_nodes)
        S = np.random.poisson(lambda_ , num_nodes)
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)

class geometric_configuration_graph(Network):
    def __init__(self , num_nodes , p):
        super().__init__(num_nodes)
        S = np.random.geometric(p , num_nodes) - 1
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)

class popular_geometric_configuration_graph(Network):
    def __init__(self , num_nodes , p, popular_nodes_num):
        super().__init__(num_nodes)
        S = np.random.geometric(p , num_nodes) - 1
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)
        # print(self.adj[0])
        popular_nodes = []
        temp = self.adj.copy()
        for i in range(popular_nodes_num):
            popular_nodes.append(max(temp, key=lambda x: len(temp[x])))
            temp.pop(max(temp, key=lambda x: len(temp[x])))
            #print(popular_nodes)
            #print(len(self.adj[0]))
        for i in popular_nodes:
            for j in popular_nodes:
                if i != j:
                    #print(i,j)
                    #print(i,j)
                    #print(self.adj[i])
                    #print(self.adj[j])
                    self.add_edge(i, j)
        
class popular_poisson_configuration_graph(Network):
    def __init__(self , num_nodes , lambda_, popular_nodes_num):
        super().__init__(num_nodes)
        S = np.random.poisson(lambda_ , num_nodes)
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)
        popular_nodes = []
        temp = self.adj.copy()
        for i in range(popular_nodes_num):
            popular_nodes.append(max(temp, key=lambda x: len(temp[x])))
            temp.pop(max(temp, key=lambda x: len(temp[x])))
        for i in popular_nodes:
            for j in popular_nodes:
                if i != j:
                    self.add_edge(i, j)


if __name__ == '__main__':
    """for i in range(11):
        edges = [random_graph(100 , i*0.1).num_edges() for _ in range(1000)]
        plt.hist(edges, bins = 50)
        plt.xlabel('Number of edges')
        plt.ylabel('Number of graphs')
        avg = np.mean(edges)
        plt.title(f'p = {np.round(i*0.1,2)}, Average = {np.round(avg)}, Variance = {np.round(np.var(edges),2)}, Expected Value = {100*99*i*0.1/2}, Expected Variance = {np.round(100*99*i*0.1*(1-i*0.1)/2,2)}')
        plt.axvline(avg , color = 'red')
        plt.savefig(f'p={i*0.1}.png',bbox_inches='tight')
        plt.close()
        #plt.show()"""
    
    """random_grah_times = np.log([timeit.timeit(lambda: random_graph(2**ii, 10/ ((2**ii)-1)) , number=100) for ii in range(6, 11)])
    lambda_graph_times = np.log([timeit.timeit(lambda: lambda_graph(2**ii, 10), number=100) for ii in range(6, 11)])
    x_axis = np.log([2**i for i in range(6, 11)])
    random_gradient, intercept = np.polyfit(x_axis, random_grah_times, 1)
    random_gradient = round(random_gradient, 2)
    lambda_gradient, intercept = np.polyfit(x_axis, lambda_graph_times, 1)
    lambda_gradient = round(lambda_gradient, 2)
    plt.plot(x_axis, random_grah_times, label=f'Random Graph, Gradient = {random_gradient}', color='blue')
    print('Random Graph Done')
    plt.plot(x_axis, lambda_graph_times, label=f'Set Number of Edges, Gradient={lambda_gradient}', color='red')
    plt.xlabel('Log Number of nodes')
    print('Lambda graph Done')
    plt.ylabel('Log Time taken')
    plt.title('Time taken for Random Graph and Lambda Graph')
    plt.legend()
    plt.show()"""

    """num_components = []
    for i in range(21):
        num_components_one = [len(lambda_graph(4096, (4096*4095*0.001*i/20/2)).find_component(1)) for _ in range(40)]
        num_components.append(np.mean(num_components_one))
    plt.plot([0.001*i/20 for i in range(21)], num_components)
    plt.axvline(1/4095, color='red')
    plt.xlabel('p')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as p changes')
    plt.show()"""

    #print(configuration_graph(10, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).degree_distributions())

    """degrees = [poisson_configuration_graph(10000, 10).degree_distributions()]
    degrees = pd.DataFrame(degrees)
    combined_degrees = degrees
    values = combined_degrees.values.flatten()
    #print(values)
    coloumns = combined_degrees.columns
    if 0 in coloumns:
        index_0 = np.where(coloumns == 0)
        value_0 = values[index_0]
        coloumns = np.delete(coloumns, index_0)
        values = np.delete(values, index_0)
        coloumns = np.insert(coloumns, 0, 0)
        values = np.insert(values, 0, value_0)

    #combined_degrees = combined_degrees.to_numpy().flatten()
    plt.plot(coloumns,values)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    
    mean = np.sum([i*j for i, j in zip(coloumns, values)])/np.sum(values)
    print(mean)
    plt.title(f'Degree distribution of a Poisson Configuration Graph, mean = {mean}')
    plt.axvline(mean, color='red')
    plt.show()"""
   
    """degrees = [geometric_configuration_graph(10000, 1/11).degree_distributions()]
    degrees = pd.DataFrame(degrees)
    combined_degrees = degrees.groupby(degrees.index).sum()
    values = combined_degrees.values.flatten()
    coloumns = combined_degrees.columns
    if 0 in coloumns:
        index_0 = np.where(coloumns == 0)
        value_0 = values[index_0]
        coloumns = np.delete(coloumns, index_0)
        values = np.delete(values, index_0)
        coloumns = np.insert(coloumns, 0, 0)
        values = np.insert(values, 0, value_0)

    #combined_degrees = combined_degrees.to_numpy().flatten()
    plt.plot(coloumns,values)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    
    mean = np.sum([i*j for i, j in zip(coloumns, values)])/np.sum(values)
    print(mean)
    plt.title(f'Degree distribution of a Geometric Configuration Graph, mean = {mean}')
    plt.axvline(mean, color='red')
    plt.show()"""

    geometric_config_graph = geometric_configuration_graph(10000, 1/11)
    poisson_config_graph = poisson_configuration_graph(10000, 10)
    """
    friend_degrees_geometric = []
    for i in range(10000):
        neighbours = np.array(list(geometric_config_graph.neighbors(np.random.randint(0, 10000))))
        if len(neighbours) > 0:
            friend_degrees_geometric.append(len(geometric_config_graph.neighbors(np.random.choice(neighbours))))
            i-=1
    friend_degrees_geometric = np.array(friend_degrees_geometric)
    plt.hist(friend_degrees_geometric, bins=20)
    mean = np.mean(friend_degrees_geometric)
    plt.axvline(mean, color='red')
    plt.title(f'Geometric friend degrees, mean = {mean}')
    plt.savefig('Geo_friend', bbox_inches='tight')
    plt.show()
    
    
    friend_degrees_poisson = []
    for i in range(10000):
        neighbours = np.array(list(poisson_config_graph.neighbors(np.random.randint(0, 10000))))
        if len(neighbours) > 0:
            friend_degrees_poisson.append(len(poisson_config_graph.neighbors(np.random.choice(neighbours))))
            i-=1
    friend_degrees_poisson = np.array(friend_degrees_poisson)
    plt.hist(friend_degrees_poisson, bins=20)
    mean = np.mean(friend_degrees_poisson)
    plt.axvline(mean, color='red')
    plt.title(f'Poisson friend degrees, mean = {mean}')
    plt.savefig('Poisson_friend', bbox_inches='tight')
    plt.show()
    """
    
    """delta_friends = geometric_config_graph.friends_degree_distribution() - geometric_config_graph.degree_distributions_individual()
    zero_index = np.where(geometric_config_graph.degree_distributions_individual() == 0)
    delta_friends = np.delete(delta_friends, zero_index)
    delta_friends[np.isnan(delta_friends)] = 0
    print(np.mean(delta_friends))
    mean = np.mean(delta_friends)
    plt.hist(delta_friends, bins=40)
    plt.axvline(mean, color='red')
    plt.title(f'Geometric Delta Configuration Graph, mean = {mean}')
    plt.savefig('Geo_delta', bbox_inches='tight')
    plt.show()

    delta_friends = poisson_config_graph.friends_degree_distribution() - poisson_config_graph.degree_distributions_individual()
    zero_index = np.where(poisson_config_graph.degree_distributions_individual() == 0)
    delta_friends = np.delete(delta_friends, zero_index)
    print(np.mean(delta_friends))
    mean = np.mean(delta_friends)
    plt.hist(delta_friends, bins=40)
    plt.axvline(mean, color='red')
    plt.title(f'Poisson Delta Configuration Graph, mean = {mean}')
    plt.savefig('Poisson_delta', bbox_inches='tight')
    plt.show()"""

    """components = []
    for i in range(11):
        num_components_one = [len(poisson_configuration_graph(10000, (2/10 * i)).find_component(1)) for _ in range(40)]
        components.append(np.mean(num_components_one))
    plt.plot([2/10 * i for i in range(11)], components)
    plt.xlabel('lambda')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as lambda changes')
    plt.savefig('Poisson_components', bbox_inches='tight')
    plt.show()

    components = []
    for i in range(11):
        num_components_one = [len(geometric_configuration_graph(10000, 1 - 2/3 * i/10).find_component(1)) for _ in range(40)]
        components.append(np.mean(num_components_one))
    
    p = np.array([1 - 2/3 * i/10 for i in range(11)])
    mean_degree = (1-p)/p
    plt.plot(mean_degree, components)
    plt.xlabel('Mean Degree')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as mean degree changes')
    plt.savefig('Geometric_components', bbox_inches='tight')
    plt.show()"""

    """components = []
    for ii in range(6):
        temp = []
        for i in range(11):
            num_components_one = [len(popular_geometric_configuration_graph(10000, 1 - 2/3 * i/10, ii*3).find_component(1)) for _ in range(10)]
            temp.append(np.mean(num_components_one))
        components.append(temp)
    
    p = np.array([1 - 2/3 * i/10 for i in range(11)])
    mean_degree = (1-p)/p
    for i in range(6):
        plt.plot(mean_degree, components[i], label=f'Popular Nodes = {i*3}')
    plt.xlabel('Mean Degree')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as mean degree changes')
    plt.legend()
    plt.savefig('Geo_Popular_components', bbox_inches='tight')
    plt.show()"""

    components = []
    for ii in range(6):
        temp = []
        for i in range(11):
            num_components_one = [len(popular_poisson_configuration_graph(10000, 2/10 * i, ii*3).find_component(1)) for _ in range(40)]
            temp.append(np.mean(num_components_one))
        components.append(temp)
    for i in range(6):
        plt.plot([2/10 * ii for ii in range(11)], components[i], label=f'Popular Nodes = {i*3}')
    plt.xlabel('lambda')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as lambda changes')
    plt.legend()
    plt.savefig('Poisson_Popular_components', bbox_inches='tight')
    plt.show()








## Question 2: It is a Binomial distribution. The average number of edges is n(n-1)p/2. The variance is n(n-1)p(1-p)/2.

## Question 3: k = number of edges, P(k) = (n-1 choose k) * p^k * (1-p)^(n-1-k), E[k] = (n-1)p, Var(k) = (n-1)p(1-p).

## Question 4: G(n, p) = G(n, \lambda / n-1). p = @lambda / (n-1). E[k] = (n-1)p = \lambda, Var(k) = (n-1)p(1-p) = \lambda (n-1-\lambda)/(n-1).
## As n goes to infinity, Var(k) goes to \lambda. The expected value of the number of edges is \lambda and doesn't depend on n.
## As n goes to infinity, P(k) = Poisson(\lambda). As (n-1 choose k)(1/n-1)^k -> 1/k! as n goes to infinity and (1-p)^(n-1-k) -> e^(-\lambda) as 1+1/x -> e as x -> infinity.
## So P(k) -> e^(-\lambda) * \lambda^k / k!.

## Questiion 5: For a probabilty configuration model, the expected value of the degree of a node is the same as the expected value of the degree of a node in a Poisson configuration model.
## The expected value of the degree of a node in a Poisson configuration model is \lambda. The expected value of the degree of a node in a probability configuration model is \sum_{i=1}^{n} i * P(i).
## For friends of a node, the expected value of the degree of a friend of a node in a Poisson configuration model is \lambda. The expected value of the degree of a friend of a node in a probability configuration model is \sum_{i=1}^{n} i * P(i) / n.