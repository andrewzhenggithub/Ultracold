#!/usr/bin/env python3
"""Analytic Solution
====================

First, we want to have an idea as to what the analytic solution tells us.  The
problem is that the analytic equation that determines the energy eigenvalues
cannot be solved analytically; thus, numerical methods must be used to find the
energy eigenvalues of the system.

"""

import pathlib

import matplotlib.pyplot as pyplot
import numpy as np
import numpy.ma as ma
from scipy.special import gamma




def energy(e: float) -> float:
    """The function that determines the energy eigenvalues.

    Parameters
    ==========

    e: float

      The input energy (in arbitrary units)


    Returns
    =======

    v: float

      Value of the energy function evaluated for the given energy.

    """

    return (1/np.sqrt(2))*(gamma(-e/2+1/2)/(gamma(-e/2+3/4)))



if __name__ == "__main__":
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()

    #Question 3________________________________________________________________________________
    #This plots the analytical solution for 

    #Global quantities
    step = 10000
    ymin = -1
    ymax = 9
    x = 5
    
    #Calling functions
    energies = np.linspace(ymin,ymax,step)
    figure, en = pyplot.subplots()
    ayes= energy (energies)
    
 
    
    #axis labeling 
    en.set_title("Energy vs $a_s$")
    en.set_ylabel("Energy")
    en.set_xlabel("$a_s$")

   
    
    #We need the code to show us the what energy values satisfy the equation for a_s
    #since the function asymptotes, we need to mask the function of prefent this plotting
    #error. This plotting error occures because the plot tries to be continuous
    ayes = ma.masked_where(np.abs(ayes)>100, ayes)

    #plotting the graph
    en.plot(ayes ,energies)

    #plotting the gridlines
    en.set_xticks(np.linspace(-5, 5, 11))
    en.set_yticks(np.linspace(1, 21, 11))
    pyplot.grid(True)

    #force the axis label to zoom in
    en.set_xlim(-x,x)
    en.set_ylim(ymin,ymax)

    #save the plot in output as question 3
    figure.savefig("output/question 3 analytic solution.pdf")

    
    


       # Example Plotting
    ########################################
    # Here is a simple example of a plotting routine.

    # First, we create an array of x values, and compute the corresponding y
    # values.
    x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    y = np.sin(x) / x
    # We create the new figure with 1 subplot (the default), and store the
    # Figure and Axis object in `fig` and `ax` (allowing for their properties
    # to be changed).
    fig, ax = pyplot.subplots()
    ax.plot(x, y)
    ax.set_title("Plot of sin(x) / x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Matplotlib by default does a good job of figuring out the limits of the
    # axes; however, it can fail sometimes.  This allows you to set them
    # manually.
    ax.set_ylim([-0.5, 1.1])
    fig.savefig("output/example.pdf")




