#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019-2021 the pept developers
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


# File   : birmingham_method.py
# License: GNU v3.0
# Author : Sam Manger <s.manger@bham.ac.uk>
# Date   : 20.08.2019


import  numpy                           as      np

import  pept

from    .extensions.birmingham_method   import  birmingham_method

import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class BirminghamMethod(pept.base.LineDataFilter):
    '''YVETTA JIAHAO OH YEAH 衝衝衝 The Birmingham Method is an efficient, analytical technique for tracking
    tracers using the LoRs from PEPT data.

    Two main methods are provided: `fit_sample` for tracking a single numpy
    array of LoRs (i.e. a single sample) and `fit` which tracks all the samples
    encapsulated in a `pept.LineData` class *in parallel*.

    For the given `sample` of LoRs (a numpy.ndarray), this function minimises
    the distance between all of the LoRs, rejecting a fraction of lines that
    lie furthest away from the calculated distance. The process is repeated
    iteratively until a specified fraction (`fopt`) of the original subset of
    LORs remains.

    This class is a wrapper around the `birmingham_method` subroutine
    (implemented in C), providing tools for asynchronously tracking samples of
    LoRs. It can return `PointData` classes which can be easily manipulated and
    visualised.

    Attributes
    ----------
    fopt : float
        Floating-point number between 0 and 1, representing the target fraction
        of LoRs in a sample used to locate a tracer.

    get_used : bool, default False
        If True, attach an attribute ``._lines`` to the output PointData
        containing the sample of LoRs used (+ a column `used`).

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.scanners.ParallelScreens : Initialise a `pept.LineData` instance from
                                    parallel screens PEPT detectors.

    Examples
    --------
    A typical workflow would involve reading LoRs from a file, instantiating a
    `BirminghamMethod` class, tracking the tracer locations from the LoRs, and
    plotting them.

    >>> import pept
    >>> from pept.tracking.birmingham_method import BirminghamMethod

    >>> lors = pept.LineData(...)   # set sample_size and overlap appropriately
    >>> bham = BirminghamMethod()
    >>> locations = bham.fit(lors)  # this is a `pept.PointData` instance

    >>> grapher = PlotlyGrapher()
    >>> grapher.add_points(locations)
    >>> grapher.show()
    '''

    def __init__(self, fopt = 0.5, get_used = False):
        '''`BirminghamMethod` class constructor.

        fopt : float, default 0.5
            Float number between 0 and 1, representing the fraction of
            remaining LORs in a sample used to locate the particle.

        verbose : bool, default False
            Print extra information when initialising this class.
        '''

        # Use @fopt.setter (below) to do the relevant type-checking when
        # setting fopt (self._fopt is the internal attribute, that we only
        # access through the getter and setter of the self.fopt property).
        self.fopt = float(fopt)
        self.get_used = bool(get_used)
        print("\n____INIT______\n")



    def PEPT_PCA(self, lors):
        # ============================= ADD LABELS =============================
        lor_a_time_interval1 = pd.DataFrame(lors[:])
        # lor_a_time_interval1["label"]=[0]*2000

        # lor_a_time_interval2 = pd.DataFrame(lors[-2000:])
        # lor_a_time_interval2["label"]=[1]*2000

        # two_particle = pd.concat([lor_a_time_interval1, lor_a_time_interval2], axis=0)

        # lor_a_time_interval3 = pd.DataFrame(lors[8000:10000])
        # lor_a_time_interval3["label"]=[2]*2000

        # three_particle = pd.concat([two_particle, lor_a_time_interval3], axis=0)
        lor_3_time_interval_without_time = lor_a_time_interval1.iloc[:,1:]
        # print("\nLORS WITH LABELS: \n", lor_3_time_interval_without_time, "\n")
        # print("=" * 50)

        # labels = lor_3_time_interval_without_time["label"]
        lor_3_time_interval_without_time=lor_3_time_interval_without_time

        # ============================= PCA =============================
        # we want to reduce the dimensionality of our data to 2 dimensions for easy visualization
        pca = PCA(n_components=3)

        # fit our pca object to the scaled data
        X_pca = pca.fit_transform(lor_3_time_interval_without_time)
        # explained variance is the fraction of the total variance in the entire dataset that a principal component accounts for
        # print("\nPCA EXPLAINED: \n", pca.explained_variance_ratio_, "\n")
        # print("=" * 50)

        # ============================= 2D GRAPH =============================
        # print("\n2D GRAPH: \n")
        # import matplotlib.pyplot as plt
        # # visualize PC1 vs PC2 with color as the cluster label
        # plt.figure(figsize=(8,6))
        # # labels =range(2000)
        # plot = plt.scatter(X_pca[:,0], X_pca[:,1], #c=labels,
        #                   linewidths=1, cmap='tab10', marker='+', alpha=0.7)
        # plt.legend(*plot.legend_elements(),
        #                     loc="upper left", title="Clusters")
        # plt.xlabel('Principal component 1')
        # plt.ylabel('Principal component 2')
        # plt.grid(alpha=0.5)
        # plt.show()
        # print("=" * 50)


        # ============================= 3D GRAPH WITH LABELS =============================
        # print("\n3D GRAPH 2: \n")
        X_pcaa = np.copy(X_pca)
        # X_pcaa = np.insert(X_pcaa, 3, labels, axis = 1)

        X_pcaa = np.insert(X_pcaa, 3, np.where((X_pca[:, 2] > -30) & (X_pca[:, 2] < 30), True, False), axis = 1)
        # X_pcaa = X_pcaa[X_pcaa[:, 3] == 1]
        # labels = X_pcaa[:, 3]

        # fig = px.scatter_3d(
        #     X_pcaa, x=0, y=1, z=2, # color=labels,
        #     labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        #     # size = [0.000000000000001]*len(X_pca)
        # )
        # fig.show()  
        # print("=" * 50)

        # ============================= HISTOGRAM =============================
        print("\nHISTOGRAME: \n")

        plt.hist(X_pcaa[:, 2], bins =100)
        plt.show()
        print(len(X_pcaa),"=" * 50)

        return X_pcaa[:, 3]

        
       
        
        
    def fit_sample(self, sample):
        '''Use the Birmingham method to track a tracer location from a numpy
        array (i.e. one sample) of LoRs.

        For the given `sample` of LoRs (a numpy.ndarray), this function
        minimises the distance between all of the LoRs, rejecting a fraction of
        lines that lie furthest away from the calculated distance. The process
        is repeated iteratively until a specified fraction (`fopt`) of the
        original subset of LORs remains.

        Parameters
        ----------
        sample : (N, M>=7) numpy.ndarray
            The sample of LORs that will be clustered. Each LoR is expressed as
            a timestamps and a line defined by two points; the data columns are
            then `[time, x1, y1, z1, x2, y2, z2, extra...]`.

        Returns
        -------
        locations : numpy.ndarray or pept.PointData
            The tracked locations found.

        used : numpy.ndarray, optional
            If `get_used` is true, then also return a boolean mask of the LoRs
            used to compute the tracer location - that is, a vector of the same
            length as `sample`, containing 1 for the rows that were used, and 0
            otherwise.
            [Used for multi-particle tracking, not implemented yet].

        Raises
        ------
        ValueError
            If `sample` is not a numpy array of shape (N, M), where M >= 7.
        '''

        if not isinstance(sample, pept.LineData):
            sample = pept.LineData(sample)


        lines__ = sample.lines         ######################################

        filter_label = self.PEPT_PCA(lines__)            ######################################
        lines__ = np.insert(lines__, 7, filter_label, axis = 1)       ######################################
        lines__ = lines__[lines__[:,-1] > 0.8]       ######################################
        lines__ = lines__[:,:-1]       ######################################

        print("\n---------------", len(lines__), "---------------\n")          ######################################
        
        sample = pept.LineData(lines__)

        #         locations, used = birmingham_method(sample.lines, self.fopt)
        locations, used = birmingham_method(sample.lines, self.fopt)        ######################################

        # Propagate any LineData attributes besides `columns`
        attrs = sample.extra_attrs()

        locations = pept.PointData(
            [locations],
            columns = ["t", "x", "y", "z", "error"],
            **attrs,
        )

        # If `get_used`, also attach a `._lines` attribute with the lines used
        if self.get_used:
            locations.attrs["_lines"] = sample.copy(
                data = np.c_[sample.lines, used],
                columns = sample.columns + ["used"],
            )

        return locations
