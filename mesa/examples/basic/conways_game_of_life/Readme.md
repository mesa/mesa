# Conway's Game Of "Life"

## Summary

[The Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), also known simply as "Life", is a cellular automaton devised by the British mathematician John Horton Conway in 1970.

The "game" is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input by a human. One interacts with the Game of "Life" by creating an initial configuration and observing how it evolves, or, for advanced "players", by creating patterns with particular properties.


## How to Run

First, clone the Mesa repository:
git clone https://github.com/projectmesa/mesa.git


Navigate to the example directory:
cd mesa/mesa/examples/basic/conways_game_of_life


Install the required dependencies:
pip install -r requirements.txt


Run the interactive visualization using Solara:
solara run app.py


Optional: To run the Streamlit version instead, install Streamlit and run:

pip install streamlit
streamlit run st_app.py



## Files

* ``agents.py``: Defines the behavior of an individual cell, which can be in two states: DEAD or ALIVE.
* ``model.py``: Defines the model itself, initialized with a random configuration of alive and dead cells.
* ``app.py``: Defines an interactive visualization using solara.
* ``st_app.py``: Defines an interactive visualization using Streamlit.

## Optional

* For the streamlit version, you need to have streamlit installed (can be done via pip install streamlit)


## Further Reading
[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
