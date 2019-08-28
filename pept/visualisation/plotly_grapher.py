#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019 Andrei Leonard Nicusan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


# File   : plotly_grapher.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.08.2019


'''The *plotly_grapher* module implements Plotly-based visualisation tools
to aid PEPT data analysis and to create publication-ready figures.

The module aims to provide general classes which can
then be used in a script file as the user sees fit. For example scripts,
look at the base of the pept library.

The peptml subpackage accepts any instace of the LineData base class
and can create matplotlib- or plotly-based figures.

PEPTanalysis requires the following packages:

* **numpy**
* **math**
* **matplotlib.pyplot** and **mpl_toolkits.mplot3d** for 3D matplotlib-based plotting
* **joblib** for multithreaded operations (such as midpoints-finding)
* **tqdm** for showing progress bars
* **plotly.subplots** and **plotly.graph_objects** for plotly-based plotting
* **hdbscan** for clustering midpoints and centres
* **time** for verbose timing of operations

It was successfuly used at the University of Birmingham to analyse real
Fluorine-18 tracers in air.

If you use this package, you should cite
the following paper: [TODO: paper signature].

'''


import  math
import  time
import  numpy                                   as          np

from    plotly.subplots                         import      make_subplots
import  plotly.graph_objects                    as          go

import  pept




class PlotlyGrapher:
    # Helper class that automatically generates Plotly graphs
    # for the PEPT data

    def __init__(self, rows=1, cols=1, xlim = [0, 500],
                 ylim = [0, 500], zlim = [0, 712], subplot_titles = ['  ']):
        self.rows = rows
        self.cols = cols

        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.subplot_titles = subplot_titles
        self.subplot_titles.extend(['  '] * (rows * cols - len(subplot_titles)))


    def create_figure(self):
        # Create subplots and set limits

        specs = [[{"type": "scatter3d"}] * self.cols] * self.rows

        self.fig = make_subplots(rows = self.rows, cols = self.cols,
                    specs = specs, subplot_titles = self.subplot_titles,
                    horizontal_spacing = 0.005, vertical_spacing = 0.05)

        self.fig['layout'].update(margin = dict(l=0,r=0,b=30,t=30), showlegend = False)

        # For every subplot (scene), set axes' ratios and limits
        # Also set the y axis to point upwards
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self.rows):
            for j in range(self.cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self.cols + j + 1)

                # Justify subplot title on the left
                self.fig.layout.annotations[i * self.cols + j].update(x = (j + 0.08) / self.cols)
                self.fig['layout'][scene].update(aspectmode = 'manual',
                                                 aspectratio = {'x': 1, 'y': 1, 'z': 1},
                                                 camera = {'up': {'x': 0, 'y': 1, 'z':0},
                                                           'eye': {'x': 1, 'y': 1, 'z': 1}},
                                                 xaxis = {'range': self.xlim,
                                                          'title': {'text': "<i>x</i> (mm)"}},
                                                 yaxis = {'range': self.ylim,
                                                          'title': {'text': "<i>y</i> (mm)"}},
                                                 zaxis = {'range': self.zlim,
                                                          'title': {'text': "<i>z</i> (mm)"}}
                                                 )

        return self.fig


    def add_data_as_trace(self, data, row = 1, col = 1, size = 2, color = None):
        # Expected data row: [time, x, y, z, ...]
        if len(data) != 0:
            trace = go.Scatter3d(
                x = data[:, 1],
                y = data[:, 2],
                z = data[:, 3],
                mode = 'markers',
                marker = dict(
                    size = size,
                    color = color,
                    opacity = 0.8
                )
            )

            self.fig.add_trace(trace, row = row, col = col)


    def add_data_as_trace_colorbar(self, data, row = 1, col = 1, title_colorbar = None, size = 3):
        # Expected data row: [time, x, y, z, ...]
        if len(data) != 0:
            if title_colorbar != None:
                colorbar = dict(title= title_colorbar)
            else:
                colorbar = dict()

            trace = go.Scatter3d(
                x=data[:, 1],
                y=data[:, 2],
                z=data[:, 3],
                mode='markers',
                marker=dict(
                    size=size,
                    color=data[:, -1],   # set color to sample size
                    colorscale='Magma',     # choose a colorscale
                    colorbar=colorbar,
                    opacity=0.8
                )
            )

            self.fig.add_trace(trace, row = row, col = col)


    def add_data_as_trace_line(self, data, row = 1, col = 1):
        # Expected data row: [time, x, y, z, ...]
        if len(data) != 0:
            trace = go.Scatter3d(
                x=data[:, 1],
                y=data[:, 2],
                z=data[:, 3],
                mode='lines',
                line=dict(
                    width=4,
                )
            )

            self.fig.add_trace(trace, row = row, col = col)


    def add_data_as_trace_lines(self, data, row = 1, col = 1):
        # data is a list of lines, each defined by two points
        # data row [time, x1, y1, z1, x2, y2, z2]
        for line in data:
            self.fig.add_trace(
                go.Scatter3d(
                    x = [ line[1], line[4] ],
                    y = [ line[2], line[5] ],
                    z = [ line[3], line[6] ],
                    mode = 'lines',
                    opacity = 0.6,
                    line = dict(
                        width = 2,
                    )
                ),
                row = row,
                col = col
			)


    def add_trace(self, trace, row = 1, col = 1):
        # Add precomputed trace
        # Can accept HDBSCANclusterer.getCentresTrace() output
        self.fig.add_trace(trace, row = row, col = col)


    def add_traces(self, traces, row = 1, col = 1):
        # Add precomputed traces
        # Can accept HDBSCANclusterer.getSampleLabelsTraces() output
        if len(traces) != 0:
            self.fig.add_traces(traces, rows=[row]*len(traces), cols=[col]*len(traces))


    def show(self):
        self.fig.show()









