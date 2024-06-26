from typing import List
import numpy as np
import random
import time
import math
from typing import Tuple
import matplotlib.pyplot as plt

class Task:
    def __init__(self, id: int, tpm: List[int]) -> None:
        self.id = id
        self.tpm = tpm
    
    def __repr__(self) -> str:
        return f"id: {self.id} | {[time for time in self.tpm]}"


class Tabu:
    def __init__(self, ban: Tuple[int], ttl: int = 7):
        self.ttl = ttl
        self.ban = ban
    
    def decrease_ttl(self):
        self.ttl -= 1
        
    def is_banned(self, order) -> bool:
        index = order.index(self.ban[0])
        if index == len(order)-1:
            return False
        return self.ban[1] in order[index+1:]
        # return order[index+1] == self.ban[1]


class TabuList:
    def __init__(self):
        self.ban_list = []
    
    def add_ban(self, ban: Tabu):
        self.ban_list.append(ban)

    def decrease_ttl(self):
        for ban in self.ban_list:
            ban.decrease_ttl()
            if ban.ttl == 0:
                self.ban_list.remove(ban)

    def is_order_banned(self, order) -> bool:
        if not self.ban_list:
            return False
        
        for ban in self.ban_list:
            if ban.is_banned(order):
                return True
        
        return False
    
    def __str__(self):
        return "\n".join([f"{tabu.ban}" for tabu in self.ban_list])     
           
def calculate_time(func):
    """
        Decorator to calculate total execution time of a function.
    """
    def inner(*args, **kwargs):
        import time
        start = time.time()
        order = func(*args, **kwargs)
        end = time.time()
        totalTime = end - start
        print(f"Execution time: {totalTime:.3} s")
        return order
        
    return inner


def readData(filepath: str) -> dict[str: List[Task]]:
    """
    Reads data from a file with sections defined by "data.XXX" lines.

    Args:
        - `filepath: str` - Path to the file containing the data.

    Returns:
        - `dict` - A dictionary where keys are section names ("data.XXX") and values are lists of lines within that section.
    """
    data = {}
    current_section = None
    counter = 0
    with open(filepath, 'r') as f:
        saveData = False
        for line in f:
            line = line.strip()
            if not line:
                saveData = False
                continue    

            if line.startswith("data."):
                saveData = True
                counter = 0
                current_section = line[:-1]
                data[current_section] = []
            else:
                if current_section and saveData:
                    if counter == 0:
                        counter += 1    
                        continue
                    tpm = [int(item) for item in line.split(" ")]
                    newTask = Task(counter, tpm)
                    data[current_section].append(newTask)
                    counter += 1 
    return data


def getTotalTime(data):
    M = len(data[0].tpm)
    machine_time = np.zeros(M)
    Cmax = 0
    for task in data:
        task_frees_at = 0
        for m in range(M):
            entry_time = max(machine_time[m], task_frees_at)
            exit_time = entry_time + task.tpm[m]
            machine_time[m] = exit_time
            Cmax = task_frees_at = exit_time
    return int(Cmax)

def printOrder(order):
    print("Order: " + " ".join([str(i+1) for i in order]))


def testSolution(data, datasetName: str, func) -> None:
    data = np.asarray(data[datasetName])
    start = time.time()
    Cmax, order = func(data)
    end = time.time()
    totalTime = end - start
    print(f"{datasetName} {getTotalTime(data[order])} {totalTime:.5} s")
    printOrder(order)
    return totalTime
    
    
def testMultiple(data, func):
    total_time = 0
    for key in data:
        total_time += testSolution(data, key, func)
    print(f"Total time: {total_time} s")
    
    
def getRandomIndices(start, stop):
    f, s = 0, 0
    while f == s:
        f, s = random.randint(start, stop), random.randint(start, stop)
    return f, s

def swap(data, i, j):
    data[i], data[j] = data[j], data[i]
    return data 

def make_step(data, tabu_list: TabuList, tabu_ttl: int):
    data = np.array(data)
    order = [t.id-1 for t in data]
    minCmax = math.inf
    best_i, best_j = None, None

    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                order = swap(order, i, j)
                if tabu_list.is_order_banned(order):
                    order = swap(order, i, j) 
                    continue 
                newCmax = getTotalTime(data[order])
                if newCmax < minCmax:
                    minCmax = newCmax
                    best_i, best_j = i, j
                order = swap(order, i, j) 

    if best_j is None or best_i is None:
        return data[order], tabu_list, None

    left, right = (best_i, best_j) if best_i < best_j else (best_j, best_i)
    ban_elements = (order[left], order[right]) 
    tabu = Tabu(ban_elements, tabu_ttl)
    tabu_list.add_ban(tabu)
    
    order = swap(order, best_i, best_j)
    
    return data[order], tabu_list, minCmax
        
        
def tabu_search(data, n_iter=1000, tabu_ttl=7):
    Cmax = []
    tabu_list = TabuList()
    for i in range(n_iter):
        data, tabu_list, cmax = make_step(data, tabu_list, tabu_ttl)
        print(f"iter: {i}: {tabu_list}")
        tabu_list.decrease_ttl()
        if cmax is not None:
            Cmax.append(cmax)
    
    plt.plot(np.arange(len(Cmax)), Cmax)
    plt.show()
    return data
    

def main():
    data = readData("data/data.txt")
    data = data["data.001"]
    # print(f"Data: {data}")
    print(f"Starting Total Time: {getTotalTime(data)}")
    data = tabu_search(data, 10, 8)
    print(f"Total_time: {getTotalTime(data)}")
    
if __name__ == "__main__":
    main()