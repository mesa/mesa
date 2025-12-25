"""Test case for reproducing the infinite loop issue in Grid.select_random_empty_cell."""

import signal

from mesa.agent import Agent
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.model import Model


def test_infinite_loop_on_full_grid():
    """Test that select_random_empty_cell does not hang on a full grid."""
    # 1. Create a small 2x2 model
    model = Model()
    grid = OrthogonalMooreGrid((2, 2))

    # 2. Fill the grid completely
    print("Filling grid...")
    for cell in grid.all_cells:
        agent = Agent(model)
        cell.add_agent(agent)

    # 3. Verify grid is full
    assert len(grid.empties) == 0
    print("Grid is full.")

    # 4. Attempt to select a random empty cell
    # Set an alarm to kill the test if it hangs
    def handler(signum, frame):
        raise TimeoutError("Test timed out! Infinite loop detected.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(2)  # Set 2 second timeout

    print("Attempting to select random empty cell...")
    try:
        grid.select_random_empty_cell()
    except IndexError:
        print("Success! Caught expected IndexError (grid is full).")
    except TimeoutError:
        print("FAILURE: The function hung in an infinite loop.")
        exit(1)
    finally:
        signal.alarm(0)  # Disable alarm


if __name__ == "__main__":
    test_infinite_loop_on_full_grid()
