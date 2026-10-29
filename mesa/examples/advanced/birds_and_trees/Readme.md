# Birds and Trees

## Summary

Birds fly through the air and eat fruit from trees on the ground below. A bird flies on with a slowly changing heading until it sees a tree with fruit nearby. It then lands on that tree, eats, and the tree grows a new fruit after a while. Birds use energy as they fly, die when they run out, and split in two when they have plenty.

The model demonstrates `mesa.experimental.locations`:

- Two spaces in the same coordinates: a continuous `air` space and a `ground` grid with `cell_size` set so that it covers the same area.
- A bird is in both. Its `position` in the air is a `ContinuousLocation`, and its ground `cell` is a `CellLocation` that follows that position. Flying moves the cell below it; landing on a cell moves the bird to the center of that cell.
- Trees use a plain `CellLocation` on the same grid, and are unaffected by the birds' link between air and ground.

The visualization draws the ground grid, with each bird on the cell below it.

## How to Run

Install Mesa with recommended dependencies:

    pip install "mesa[rec]"

Then run the example:

    solara run app.py

Open the displayed local URL in your browser.

## Files

* `model.py`: the model, with the air and the ground
* `agents.py`: the `Bird` and `Tree` agents
* `app.py`: Solara based visualization
