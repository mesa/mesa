import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from mesa.examples.basic.conways_game_of_life.model import ConwaysGameOfLife

model = st.title("Conway's Game of Life")
num_ticks = st.slider("Select number of Steps", min_value=1, max_value=100, value=50)
height = st.slider("Select Grid Height", min_value=10, max_value=100, step=10, value=15)
width = st.slider("Select Grid Width", min_value=10, max_value=100, step=10, value=20)
model = ConwaysGameOfLife(height, width)

col1, col2, col3 = st.columns(3)
status_text = st.empty()
# step_mode = st.checkbox('Run Step-by-Step')
run = st.button("Run Simulation")


if run:
    tick = time.time()
    step = 0
    # init grid
    df_grid = pd.DataFrame()
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for x in range(width):
        for y in range(height):
            df_grid = pd.concat(
                [df_grid, pd.DataFrame({"x": [x], "y": [y], "state": [0]})],
                ignore_index=True,
            )

    heatmap = (
        alt.Chart(df_grid)
        .mark_point(size=100)
        .encode(x="x", y="y", color=alt.Color("state"))
        .interactive()
        .properties(width=800, height=600)
    )

    # init progress bar
    my_bar = st.progress(0, text="Simulation Progress")  # progress
    placeholder = st.empty()
    st.subheader("Agent Grid")
    chart = st.altair_chart(heatmap, use_container_width=True)
    color_scale = alt.Scale(domain=[0, 1], range=["red", "yellow"])
    for i in range(num_ticks):
        model.step()
        my_bar.progress((i / num_ticks), text="Simulation progress")
        placeholder.text(f"Step = {i}")
        for contents, (x, y) in model.grid.coord_iter():
            selected_row = df_grid[(df_grid["x"] == x) & (df_grid["y"] == y)]
            df_grid.loc[selected_row.index, "state"] = contents.state

        heatmap = (
            alt.Chart(df_grid)
            .mark_circle(size=100)
            .encode(x="x", y="y", color=alt.Color("state", scale=color_scale))
            .interactive()
            .properties(width=800, height=600)
        )
        chart.altair_chart(heatmap)

        time.sleep(0.1)

    tock = time.time()
    st.success(f"Simulation completed in {tock - tick:.2f} secs")
