import heapq

from typing import Any, List, Union


class PriorityQueue:
    heap: List[Any]  # Heap data structure
    entry_finder: dict  # Dictionary mapping elements to entries
    counter: int  # Unique sequence count -> Queue position

    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0

    __repr__ = __str__ = lambda self: str(
        [(x[-1], x[0]) for x in self.heap if x[-1] is not None]
    )

    def __contains__(self, element) -> bool:
        return element in self.entry_finder

    def __len__(self) -> int:
        return len(self.entry_finder)

    def insert(self, element: Any, cost: Union[int, float] = 0.0):
        """This function will insert an element into the priority queue.

        Args:
            element (Any): Element to be inserted into the priority queue.
            cost (Number): Cost of the element. Defaults to 0.0.
        """
        entry = [cost, self.counter, element]
        self.entry_finder[element] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def append(self, element: Any, cost: Union[int, float] = 0.0):
        self.insert(element, cost)

    def pop(self):
        """This function will pop the element with the lowest cost or which is longest in the queue (if costs are equal)."""
        while self.heap:
            _, _, element = heapq.heappop(self.heap)
            if element is not None:
                del self.entry_finder[element]
                return element
        raise IndexError("Priority queue is empty.")

    def update_cost(self, element: Any, new_cost: Union[int, float]):
        """This function does update the cost of a given element.

        Args:
            element (Any): Element which cost is to be updated.
            new_cost (Union[int, float]): New cost of the element.
        """
        if element in self.entry_finder:
            entry = self.entry_finder[element]
            entry[0] = new_cost  # Update the cost in the entry
            heapq.heapify(self.heap)  # Reorder the heap based on the updated cost
            self.entry_finder[element] = entry  # Update the entry in the dictionary
    def is_empty(self):
        """This function checks if the priority queue is empty."""
        return len(self.entry_finder) == 0
