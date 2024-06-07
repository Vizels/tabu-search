"Tabu Search algorithm implementation." ""

import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


class Task:
    "Task representation with id and time per machine."

    def __init__(self, id: int, tpm: List[int]) -> None:
        self.id = id
        self.tpm = tpm

    def __repr__(self) -> str:
        return f"\nid: {self.id} | {list(self.tpm)}"


class Tabu:
    "Tabu representation with banned pair of tasks and time to live."

    def __init__(self, banned_pair: Tuple[Task], ttl: int = 7):
        self.ttl = ttl
        self.banned_pair = banned_pair

    def decrease_ttl(self):
        "Decrease tabu's time to live."
        self.ttl -= 1

    def is_banned(self, data) -> bool:
        """
        Check if the pair of tasks is banned
        (right is not allowed to be after left)
        """
        left_index = data.index(self.banned_pair[0])
        if left_index == len(data) - 1:
            return False
        return self.banned_pair[1] in data[left_index + 1 :]

    def __str__(self):
        return f"{self.banned_pair[0].id} -> {self.banned_pair[1].id}"


class TabuList:
    "List of Tabu objects."

    def __init__(self):
        self.ban_list = []

    def add_ban(self, tabu: Tabu):
        "Add a new tabu to the list."
        self.ban_list.append(tabu)

    def decrease_ttl(self):
        "Decrease all tabus' time to live."
        for tabu in self.ban_list:
            tabu.decrease_ttl()
            if tabu.ttl == 0:
                self.ban_list.remove(tabu)

    def is_dataorder_banned(self, data) -> bool:
        "Check if the data order is banned."
        if not self.ban_list:
            return False

        for tabu in self.ban_list:
            if tabu.is_banned(data):
                return True

        return False

    def __str__(self):
        return (
            "\n".join([f"{tabu}, ttl: {tabu.ttl}" for tabu in self.ban_list])
            + "\n---------------"
        )


def read_data(filepath: str) -> dict[str : List[Task]]:
    """
    Reads data from a file with sections defined by "data.XXX" lines.

    Args:
        - `filepath: str` - Path to the file containing the data.

    Returns:
        - `dict` - A dictionary where keys are section names ("data.XXX")
        and values are lists of lines within that section.
    """
    data = {}
    current_section = None
    counter = 0
    with open(filepath, "r") as f:
        save_data = False
        for line in f:
            line = line.strip()
            if not line:
                save_data = False
                continue

            if line.startswith("data."):
                save_data = True
                counter = 0
                current_section = line[:-1]
                data[current_section] = []
            else:
                if current_section and save_data:
                    if counter == 0:
                        counter += 1
                        continue
                    tpm = [int(item) for item in line.split(" ")]
                    new_task = Task(counter, tpm)
                    data[current_section].append(new_task)
                    counter += 1
    return data


def getTotalTime(data):
    "Calculate the total execution time of the given data."
    M = len(data[0].tpm)
    machine_time = np.zeros(M)
    cmax = 0
    for task in data:
        task_frees_at = 0
        for m in range(M):
            entry_time = max(machine_time[m], task_frees_at)
            exit_time = entry_time + task.tpm[m]
            machine_time[m] = exit_time
            cmax = task_frees_at = exit_time
    return int(cmax)


def swap(data, task1: Task, task2: Task):
    "Swap two tasks in the data."
    data = data.copy()
    index1, index2 = data.index(task1), data.index(task2)
    data[index1], data[index2] = data[index2], data[index1]
    return data


def make_step(data, tabu_list: TabuList, tabu_ttl=7):
    "Make a step in the tabu search algorithm."
    min_cmax = math.inf
    best_i, best_j = None, None

    ctr = 0
    for i in data:
        for j in data[data.index(i) :]:
            if i != j:
                ctr += 1
                swapped_data = swap(data, i, j)
                new_cmax = getTotalTime(swapped_data)
                if not tabu_list.is_dataorder_banned(swapped_data):
                    if new_cmax < min_cmax:
                        min_cmax = new_cmax
                        best_i, best_j = i, j

    tabu_list.decrease_ttl()

    if best_i is not None and best_j is not None:
        new_ban = Tabu((best_i, best_j), tabu_ttl)
        tabu_list.add_ban(new_ban)
        data = swap(data, best_i, best_j)
        return data, tabu_list, min_cmax

    return data, tabu_list, None


def tabu_search(data, n_iter=1000, tabu_ttl=7, use_best=False, show_plot=False):
    "Tabu search algorithm implementation."
    init_cmax = getTotalTime(data)
    cmax_history = [init_cmax]
    best_cmax = math.inf

    tabu_list = TabuList()
    for i in range(n_iter):
        data, tabu_list, cmax = make_step(data, tabu_list, tabu_ttl)
        # print(f"iter: {i}: \n{tabu_list}")
        if cmax is not None:
            cmax_history.append(cmax)
            if use_best:
                if cmax < best_cmax:
                    best_data = data
                    best_cmax = cmax

    if show_plot:
        plt.plot(np.arange(len(cmax_history)), cmax_history)
        plt.show()

    return best_data if use_best else data


def test_different_parameters(
    data: str, start: int, stop: int, step: int, parameter_type: str, use_best=False
):
    "Test the algorithm with different parameters."
    data = read_data("data/data.txt")
    data = data["data.001"]
    parameter_history = [i for i in range(start, stop, step)]
    Cmax_history = []
    for i in parameter_history:
        if parameter_type == "tabu_ttl":
            data = tabu_search(
                data, 500, tabu_ttl=i, show_plot=False, use_best=use_best
            )
            plt.xlabel("Tabu TTL")
            plt.ylabel("Cmax")
        elif parameter_type == "n_iter":
            data = tabu_search(data, n_iter=i, show_plot=False, use_best=use_best)
            plt.xlabel("Number of iterations")
            plt.ylabel("Cmax")
        Cmax_history.append(getTotalTime(data))

    plt.plot(parameter_history, Cmax_history)
    plt.show()


def main():
    data = read_data("data/data.txt")
    data = data["data.040"]
    # print(f"Data: {data}")
    print(f"Starting Total Time: {getTotalTime(data)}")
    data = tabu_search(data, 10000, tabu_ttl=10, show_plot=True, use_best=True)
    print(f"Total_time: {getTotalTime(data)}")


if __name__ == "__main__":
    #main()
    test_different_parameters(
        data="data.010",
        start=1,
        stop=15,
        step=1,
        parameter_type="tabu_ttl",
    )
