from typing import List, Tuple
import numpy as np
import math
import matplotlib.pyplot as plt

class Task:
    def __init__(self, id: int, tpm: List[int]) -> None:
        self.id = id
        self.tpm = tpm
    
    def __repr__(self) -> str:
        return f"\nid: {self.id} | {[time for time in self.tpm]}"


class Tabu:
    def __init__(self, banned_pair: Tuple[Task], ttl: int = 7):
        self.ttl = ttl
        self.banned_pair = banned_pair
    
    def decrease_ttl(self):
        self.ttl -= 1
        
    def is_banned(self, data) -> bool:
        '''
        Check if the pair of tasks is banned
        (right is not allowed to be after left)
        '''
        left_index = data.index(self.banned_pair[0])
        if left_index == len(data)-1:
            return False
        return self.banned_pair[1] in data[left_index+1:]

    def __str__(self):
        return f"{self.banned_pair[0].id} -> {self.banned_pair[1].id}"

class TabuList:
    def __init__(self):
        self.ban_list = []
    
    def add_ban(self, banned_tabu: Tabu):
        self.ban_list.append(banned_tabu)

    def decrease_ttl(self):
        for banned_tabu in self.ban_list:
            banned_tabu.decrease_ttl()
            if banned_tabu.ttl == 0:
                self.ban_list.remove(banned_tabu)

    def is_dataorder_banned(self, data) -> bool:
        if not self.ban_list:
            return False
        
        for banned_tabu in self.ban_list:
            if banned_tabu.is_banned(data):
                return True
        
        return False
    
    def __str__(self):
        return "\n".join([f"{tabu}, ttl: {tabu.ttl}" for tabu in self.ban_list])+"\n---------------"     



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

def swap(data, task1: Task, task2: Task):
    data = data.copy()
    index1, index2 = data.index(task1), data.index(task2)
    data[index1], data[index2] = data[index2], data[index1]
    return data 


def make_step(data, tabu_list: TabuList, tabu_ttl=7):
    min_Cmax = math.inf
    best_i, best_j = None, None
    

    ctr = 0
    for i in data:
        for j in data[data.index(i):]:
            if i != j:
                ctr +=1
                swapped_data = swap(data, i, j)
                new_Cmax = getTotalTime(swapped_data)
                if not tabu_list.is_dataorder_banned(swapped_data):
                    if new_Cmax < min_Cmax:
                        min_Cmax = new_Cmax
                        best_i, best_j = i, j
    
    tabu_list.decrease_ttl()


    if best_i is not None and best_j is not None:
        new_ban = Tabu((best_i, best_j), tabu_ttl)
        tabu_list.add_ban(new_ban)
        data = swap(data, best_i, best_j)
        return data, tabu_list, min_Cmax
    
    print("NO WAY")
    
    return data, tabu_list, None
          
def tabu_search(data, n_iter=1000, tabu_ttl=7):
    init_Cmax = getTotalTime(data)
    Cmax_history = [init_Cmax]

    tabu_list = TabuList()
    for i in range(n_iter):
        data, tabu_list, cmax = make_step(data, tabu_list, tabu_ttl)
        print(f"iter: {i}: \n{tabu_list}")
        if cmax is not None:
            Cmax_history.append(cmax)
    
    plt.plot(np.arange(len(Cmax_history)), Cmax_history)
    plt.show()
    return data

def main():
    data = readData("data/data.txt")
    data = data["data.001"]
    # print(f"Data: {data}")
    print(f"Starting Total Time: {getTotalTime(data)}")
    data = tabu_search(data, 20, tabu_ttl=10)
    print(f"Total_time: {getTotalTime(data)}")
    
if __name__ == "__main__":
    main()