# _*_ coding: utf-8 _*_
"""
Created on 08/07/2022
	Functions to graph Henderson-Hasselbach in function
	absorbance
@author: ADOB

require matplotlib, sympy, numpy, scipy
execute: pip install matplotlib sympy numpy scipy
"""

from mycontrollers.math.linearmath import LSOnlyGraph
from mycontrollers.math.linearmath import LinearSolveComp
from mycontrollers.math.linearmath import LinearSolve
from mycontrollers.math.linearmath import LSOGComp
from myviews.graphmaker import Graphmaker

#==========================================================================
## Molar fraction aqua and molar volume
y_ph = [1.09, 2.35, 3.52, 4.41, 5.66, 6.26, 7.2, 8.4, 10.87, 12.35]
x_log_data = [
-3.470190837, 
-2.856231623, 
-2.893768783, 
-2.916854526,
-1.808580509, 
-1.264988404,
-0.483469106,
0.252734286,
0.490560347,
0.454048339]

y_ph1 = [3.52, 4.41, 5.66, 6.26, 7.2, 8.4]
x_log_data1 = [
-2.893768783,
-2.916854526,
-1.808580509, 
-1.264988404,
-0.483469106,
0.252734286]

mygraph = Graphmaker()

#==========================================================================

# Plot graph and linearRegression type Time vs Temperature
def Makegrap():

	## pH vs log(A)
	mygraph.Graph(
		r"pH = f($\bar{A}$)",
		r"$log(\frac{A_{\lambda{2}}A_{\lambda{1, ácido}}}{A_{\lambda{1}}A_{\lambda{2, básico}}})$",
		r"pH",
		"pHvslog.png", x_log_data, y_ph,
		r"pH = f[$log(\frac{A_{\lambda{2}}A_{\lambda{1, ácido}}}{A_{\lambda{1}}A_{\lambda{2, básico}}})$]"
	)

	## Linealización
	LSOnlyGraph(
	x_log_data,
	y_ph,
	r"pH = f($\bar{A}$)",
	r"$log(\frac{A_{\lambda{2}}A_{\lambda{1, ácido}}}{A_{\lambda{1}}A_{\lambda{2, básico}}})$",
	r"pH",
	"linealizacionOnly_pHvslog.png",
	6
	)

	## Linealización
	LinearSolveComp(
	x_log_data,
	y_ph,
	r"pH = f($\bar{A}$)",
	r"$log(\frac{A_{\lambda{2}}A_{\lambda{1, ácido}}}{A_{\lambda{1}}A_{\lambda{2, básico}}})$",
	r"pH",
	"linealizacion_pHvslog.png",
	6
	)

	LSOGComp(
	x_log_data1,
	y_ph1,
	r"pH = f($\bar{A}$)",
	r"$log(\frac{A_{\lambda{2}}A_{\lambda{1, ácido}}}{A_{\lambda{1}}A_{\lambda{2, básico}}})$",
	r"pH",
	"LO_pHvslog.png",
	6
	)

if __name__ == "__main__":
	Makegrap()