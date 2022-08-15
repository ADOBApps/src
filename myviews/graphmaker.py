# _*_ coding: utf-8 _*_
"""
Created on 28/07/2022
	Adaptation of salt class
@author: ADOB

require matplotlib, sympy, numpy, scipy
execute: pip install matplotlib sympy numpy scipy

"""

import matplotlib.pyplot as plt
import numpy as np

class Graphmaker:

	# Initial function
	def __init__(self):
		print("Calling constructor")
		class_name = self.__class__.__name__
		print(class_name, "Ready")

		# set params characteristics at all plots and subplots for implements latext
		params = {'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
		plt.rcParams.update(params)

	# Destroyer function
	def __del__ (self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")

	# Personalization
	def Graph(self, curve_name, _xlabel, _ylabel, graph_name, _time, _temp1, _latex):
		# Create figure and axes
		fig = plt.figure()
		fig.clf()
		ax = fig.add_subplot(1,1,1)

		# Set title
		ax.set_title(curve_name)

		# Create the plot
		ax.set_xlabel(_xlabel)
		ax.set_ylabel(_ylabel)
		ax.plot(_time, _temp1, label=_latex)
		ax.plot(_time, _temp1, 'bo')

		# Show the major and minor grid lines
		ax.grid(visible=True, which='major', color='#666666', linestyle='-')
		ax.minorticks_on()
		ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		## Legend
		ax.legend(prop={'size': 10}, loc="lower right")

		# Save figure
		fig.savefig(graph_name)

		# Show plot
		plt.show()
