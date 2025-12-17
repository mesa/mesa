# Run with streamlit run st_app.py

import time

import altair as alt
import pandas as pd
import streamlit as st
from model import BoltzmannWealth

model = st.title("Boltzman Wealth Model")
num_agents = st.slider(
    "Choose how many agents to include in the model",
    min_value=1,
    max_value=100,
    value=50,
)
num_ticks = st.slider(
    "Select number of Simulation Runs", min_value=1, max_value=100, value=50
)
height = st.slider("Select Grid Height", min_value=10, max_value=100, step=10, value=15)
width = st.slider("Select Grid Width", min_value=10, max_value=100, step=10, value=20)
model = BoltzmannWealth(num_agents, height, width)


status_text = st.empty()
run = st.button("Run Simulation")


if run:
    tick = time.time()
    step = 0
    # init grid
    df_grid = pd.DataFrame()
    df_gini = pd.DataFrame({"step": [0], "gini": [-1]})
    for x in range(width):
        for y in range(height):
            df_grid = pd.concat(
                [df_grid, pd.DataFrame({"x": [x], "y": [y], "agent_count": 0})],
                ignore_index=True,
            )

    heatmap = (
        alt.Chart(df_grid)
        .mark_point(size=100)
        .encode(x="x", y="y", color=alt.Color("agent_count"))
        .interactive()
        .properties(width=800, height=600)
    )

    line = (
        alt.Chart(df_gini)
        .mark_line(point=True)
        .encode(x="step", y="gini")
        .properties(width=800, height=600)
    )

    # init progress bar
    my_bar = st.progress(0, text="Simulation Progress")  # progress
    placeholder = st.empty()
    st.subheader("Agent Grid")
    chart = st.altair_chart(heatmap)
    st.subheader("Gini Values")
    line_chart = st.altair_chart(line)

    color_scale = alt.Scale(
        domain=[0, 1, 2, 3, 4], range=["red", "cyan", "white", "white", "blue"]
    )
    for i in range(num_ticks):
        model.step()
        my_bar.progress((i / num_ticks), text="Simulation progress")
        placeholder.text(f"Step = {i}")
        for cell in model.grid.coord_iter():
            cell_content, (x, y) = cell
            agent_count = len(cell_content)
            selected_row = df_grid[(df_grid["x"] == x) & (df_grid["y"] == y)]
            df_grid.loc[selected_row.index, "agent_count"] = agent_count

        df_gini = pd.concat(
            [
                df_gini,
                pd.DataFrame(
                    {"step": [i], "gini": [model.datacollector.model_vars["Gini"][i]]}
                ),
            ]
        )
        # st.table(df_grid)
        heatmap = (
            alt.Chart(df_grid)
            .mark_circle(size=100)
            .encode(x="x", y="y", color=alt.Color("agent_count", scale=color_scale))
            .interactive()
            .properties(width=800, height=600)
        )
        chart.altair_chart(heatmap)

        line = (
            alt.Chart(df_gini)
            .mark_line(point=True)
            .encode(x="step", y="gini")
            .properties(width=800, height=600)
        )
        line_chart.altair_chart(line)

        time.sleep(0.01)

    tock = time.time()
    st.success(f"Simulation completed in {tock - tick:.2f} secs")

    # st.subheader('Agent Grid')
    # fig = px.imshow(agent_counts,labels={'color':'Agent Count'})
    # st.plotly_chart(fig)
    # st.subheader('Gini value over sim ticks (Plotly)')
    # chart = st.line_chart(model.datacollector.model_vars['Gini'])
