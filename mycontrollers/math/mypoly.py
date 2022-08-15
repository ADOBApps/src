# _*_ coding: utf-8 _*_
"""
Created on 09/08/2022
	Class to work with polynomials
@author: ADOB

require matplotlib, numpy, scipy, seaborn, sklearn, pandas, statsmodels
execute: pip install matplotlib numpy scipy seaborn sklearn pandas statsmodels
"""
# Graph
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
from tkinter import messagebox as mb
# Data treatment
# ==============================================================================
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
# Models
# ==============================================================================
from scipy.stats import pearsonr

# Configuración warnings
# ==============================================================================
import warnings


class ADOBpoly:
	def __init__(self, x_data, y_data, endorder, figname, legendsize):
		print("Calling constructor")
		class_name = self.__class__.__name__
		print(class_name, "Ready")
		warnings.filterwarnings('ignore')

		# Data entry verify
		if(endorder<3):
			mb.showinfo("Order", "Order lass than allowed")
			endorder = 3
		elif(endorder==3):
			mb.showinfo("Order", f"Max {endorder} can generate issues")
			endorder = 3
		else:
			endorder = endorder

		#Data manage
		data = pd.DataFrame({'varindependent': x_data, 'vardependent': y_data})
		data.head(3)
		print("")
		print(data)
		print("")

		#Graph config
		# ==============================================================================
		# set params characteristics at all plots and subplots for implements latext
		params = {'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
		plt.rcParams.update(params)

		# Initialise the figure and a subplot axes. Each subplot sharing (showing) the
		# same range of values for the x and y axis in the plots
		fig, ax = plt.subplots(1, 1,figsize=(6, 4), sharex=True, sharey=True)

		data.plot(
			x = 'varindependent',
			y = 'vardependent',
			c = 'firebrick',
			kind = "scatter",
			ax = ax
		)
		# Show the minor grid lines with very faint and almost transparent grey lines
		ax.grid(visible=True, which='major', color='#666666', linestyle='-')
		ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		ax.minorticks_on()

		# x values for model
		xstart = min(x_data)-1
		xstop = max(x_data)+1
		increment = 0.01
		xmodel = np.arange(xstart, xstop, increment)

		startorder = 2

		for model_order in range(startorder, endorder, 1):
			# Finding the moderl
			p = np.polyfit(x_data, y_data, model_order)

		# Plot the model
		ymodel = np.polyval(p, xmodel)
		ax.plot(xmodel, ymodel, label=f"$y=x^{model_order}$")
		ax.legend(fontsize=legendsize);

		plt.show()

	def __del__(self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")