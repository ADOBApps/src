# _*_ coding: utf-8 _*_
"""
Created on 08/07/2022
	Class to graph linearRegressions
@author: ADOB

require matplotlib, numpy, scipy, seaborn, sklearn, pandas, statsmodels
execute: pip install matplotlib numpy scipy seaborn sklearn pandas statsmodels
"""

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Configuración warnings
# ==============================================================================
import warnings

## Graph linear regression without confidene interval (CI) curves
class LSOnlyGraph:

	# Constructor function
	def __init__ (self, xs, ys, title, xlabel, ylabel, figname, legendsize):
		warnings.filterwarnings('ignore')

		print("Calling constructor")
		class_name = self.__class__.__name__
		print(class_name, "Ready")
		
		varindependent = xs
		vardependent = ys
		
		datos = pd.DataFrame({'varindependent': varindependent, 'vardependent': vardependent})
		datos.head(3)
		print("")
		print(datos)
		print("")

		# Gráfico
		# ==============================================================================
		# set params characteristics at all plots and subplots for implements latext
		params = {'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
		plt.rcParams.update(params)

		# Initialise the figure and a subplot axes. Each subplot sharing (showing) the
		# same range of values for the x and y axis in the plots
		fig, ax = plt.subplots(2, 1,figsize=(6, 6), sharex=True, sharey=True)

		# Set the title for the figure
		#fig.suptitle("Linealización", fontsize=10)

		datos.plot(
			x = 'varindependent',
			y = 'vardependent',
			c = 'firebrick',
			kind = "scatter",
			ax = ax[0]
		)
		ax[0].set_title(title);
		ax[0].set_xlabel(xlabel)
		ax[0].set_ylabel(ylabel)

		# Show the major grid lines with dark grey lines
		ax[0].grid(visible=True, which='major', color='#666666', linestyle='-')
		ax[0].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		ax[0].minorticks_on()

		# Show the minor grid lines with very faint and almost transparent grey lines
		ax[1].grid(visible=True, which='major', color='#666666', linestyle='-')
		ax[1].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		ax[1].minorticks_on()

		# Correlación lineal entre las dos variables
		# ==============================================================================
		corr_test = pearsonr(x = datos['varindependent'], y =  datos['vardependent'])
		print("Coeficiente de correlación de Pearson: ", corr_test[0])
		print("P-value: ", corr_test[1])

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)

		# Creación del modelo
		# ==============================================================================
		modelo = LinearRegression()
		modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

		# Información del modelo
		# ==============================================================================
		print("Intercept (b):", modelo.intercept_)
		self.b1 = modelo.intercept_
		self.m1 = list(zip(X.columns, modelo.coef_.flatten(), ))
		print("Coeficiente (m):", list(zip(X.columns, modelo.coef_.flatten(), )))
		print("Coeficiente de determinación R^2:", modelo.score(X, y))

		# Error de test del modelo 
		# ==============================================================================
		predicciones_modelo1 = modelo.predict(X = X_test)
		print(predicciones_modelo1[0:3,])

		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_modelo1,
			squared = False
		)

		print("")
		print(f"El error (rmse) de test es: {rmse}")
		print("")
		print("")

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)
		# Creación del modelo utilizando matrices como en scikitlearn
		# ==============================================================================
		# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
		X_train = sm.add_constant(X_train, prepend=True)
		modelo = sm.OLS(endog=y_train, exog=X_train,)
		modelo = modelo.fit()
		print(modelo.summary())

		# Intervalos de confianza para los coeficientes del modelo
		# ==============================================================================
		modelo.conf_int(alpha=0.05)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones.head(4)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones['x'] = X_train[:, 1]
		predicciones['y'] = y_train
		predicciones = predicciones.sort_values('x')

		# Error de test del modelo 
		# ==============================================================================
		X_test = sm.add_constant(X_test, prepend=True)
		predicciones_error = modelo.predict(exog = X_test)
		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_error,
			squared = False
			)
		print(f"El error (rmse) de test es: {rmse}")

		# Imprimimos las regresiones
		# ==============================================================================
		print("")
		print("Regresión Linear 1:", f"{self.m1[0][1]}x+{self.b1[0]}")
		print("")
		print("")

		# Gráfico del modelo
		# ==============================================================================
		#fig, ax = plt.subplots(figsize=(6, 3.84))
		
		ax[1].set_title(f"{title} Linealizada");
		ax[1].set_xlabel(xlabel)
		ax[1].set_ylabel(ylabel)

		ax[1].scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
		ax[1].plot(predicciones['x'], predicciones["mean"], linestyle='-', label=f"{self.m1[0][1]}x + {self.b1[0]}")
		ax[1].fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
		ax[1].legend(fontsize=legendsize);

		plt.savefig(figname)
		plt.show()
	
	# Destroyer function
	def __del__ (self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")

## Graph curve and linear regression in same frame with CI's curve
class LinearSolveComp:

	# Constructor function
	def __init__ (self, xs, ys, title, xlabel, ylabel, figname, legendsize):
		warnings.filterwarnings('ignore')

		print("Calling constructor")
		class_name = self.__class__.__name__
		print(class_name, "Ready")
		
		varindependent = xs
		vardependent = ys
		
		datos = pd.DataFrame({'varindependent': varindependent, 'vardependent': vardependent})
		datos.head(3)
		print("")
		print(datos)
		print("")

		# Gráfico
		# ==============================================================================
		# set params characteristics at all plots and subplots for implements latext
		params = {'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
		plt.rcParams.update(params)

		# Initialise the figure and a subplot axes. Each subplot sharing (showing) the
		# same range of values for the x and y axis in the plots
		fig, ax = plt.subplots(2, 1,figsize=(6, 6), sharex=True, sharey=True)

		# Set the title for the figure
		#fig.suptitle("Linealización", fontsize=10)

		datos.plot(
			x = 'varindependent',
			y = 'vardependent',
			c = 'firebrick',
			kind = "scatter",
			ax = ax[0]
		)
		ax[0].set_title(title);
		ax[0].set_xlabel(xlabel)
		ax[0].set_ylabel(ylabel)

		# Show the major grid lines with dark grey lines
		ax[0].grid(visible=True, which='major', color='#666666', linestyle='-')
		ax[0].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		ax[0].minorticks_on()

		# Show the minor grid lines with very faint and almost transparent grey lines
		ax[1].grid(visible=True, which='major', color='#666666', linestyle='-')
		ax[1].grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
		ax[1].minorticks_on()

		# Correlación lineal entre las dos variables
		# ==============================================================================
		corr_test = pearsonr(x = datos['varindependent'], y =  datos['vardependent'])
		print("Coeficiente de correlación de Pearson: ", corr_test[0])
		print("P-value: ", corr_test[1])

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)

		# Creación del modelo
		# ==============================================================================
		modelo = LinearRegression()
		modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

		# Información del modelo
		# ==============================================================================
		print("Intercept (b):", modelo.intercept_)
		self.b1 = modelo.intercept_
		self.m1 = list(zip(X.columns, modelo.coef_.flatten(), ))
		print("Coeficiente (m):", list(zip(X.columns, modelo.coef_.flatten(), )))
		print("Coeficiente de determinación R^2:", modelo.score(X, y))

		# Error de test del modelo 
		# ==============================================================================
		predicciones_modelo1 = modelo.predict(X = X_test)
		print(predicciones_modelo1[0:3,])

		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_modelo1,
			squared = False
		)

		print("")
		print(f"El error (rmse) de test es: {rmse}")
		print("")
		print("")

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)
		# Creación del modelo utilizando matrices como en scikitlearn
		# ==============================================================================
		# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
		X_train = sm.add_constant(X_train, prepend=True)
		modelo = sm.OLS(endog=y_train, exog=X_train,)
		modelo = modelo.fit()
		print(modelo.summary())

		# Intervalos de confianza para los coeficientes del modelo
		# ==============================================================================
		modelo.conf_int(alpha=0.05)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones.head(4)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones['x'] = X_train[:, 1]
		predicciones['y'] = y_train
		predicciones = predicciones.sort_values('x')

		# Error de test del modelo 
		# ==============================================================================
		X_test = sm.add_constant(X_test, prepend=True)
		predicciones_error = modelo.predict(exog = X_test)
		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_error,
			squared = False
			)
		print(f"El error (rmse) de test es: {rmse}")

		# Imprimimos las regresiones
		# ==============================================================================
		print("")
		print("Regresión Linear 1:", f"{self.m1[0][1]}x+{self.b1[0]}")
		print("")
		print("")

		# Gráfico del modelo
		# ==============================================================================
		#fig, ax = plt.subplots(figsize=(6, 3.84))
		
		ax[1].set_title(f"{title} Linealizada");
		ax[1].set_xlabel(xlabel)
		ax[1].set_ylabel(ylabel)

		ax[1].scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
		ax[1].plot(predicciones['x'], predicciones["mean"], linestyle='-', label=f"{self.m1[0][1]}x + {self.b1[0]}")
		ax[1].plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
		ax[1].plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
		ax[1].fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
		ax[1].legend(fontsize=legendsize);

		plt.savefig(figname)
		plt.show()
	
	# Destroyer function
	def __del__ (self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")

## Graph linear regression with CI's curve
class LinearSolve:

	# Constructor function
	def __init__ (self, xs, ys, title, xlabel, ylabel, figname, legendsize):
		warnings.filterwarnings('ignore')

		print("Calling constructor")
		class_name = self.__class__.__name__
		print(class_name, "Ready")
		
		varindependent = xs
		vardependent = ys
		
		datos = pd.DataFrame({'varindependent': varindependent, 'vardependent': vardependent})
		datos.head(3)
		print("")
		print(datos)
		print("")

		# Gráfico
		# ==============================================================================
		# set params characteristics at all plots and subplots for implements latext
		params = {'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
		plt.rcParams.update(params)

		# Initialise the figure and a subplot axes. Each subplot sharing (showing) the
		# same range of values for the x and y axis in the plots
		fig, ax = plt.subplots(1, 1,figsize=(6, 6), sharex=True, sharey=True)

		# Set the title for the figure
		#fig.suptitle("Linealización", fontsize=10)

		datos.plot(
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

		# Correlación lineal entre las dos variables
		# ==============================================================================
		corr_test = pearsonr(x = datos['varindependent'], y =  datos['vardependent'])
		print("Coeficiente de correlación de Pearson: ", corr_test[0])
		print("P-value: ", corr_test[1])

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)

		# Creación del modelo
		# ==============================================================================
		modelo = LinearRegression()
		modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

		# Información del modelo
		# ==============================================================================
		print("Intercept (b):", modelo.intercept_)
		self.b1 = modelo.intercept_
		self.m1 = list(zip(X.columns, modelo.coef_.flatten(), ))
		print("Coeficiente (m):", list(zip(X.columns, modelo.coef_.flatten(), )))
		print("Coeficiente de determinación R^2:", modelo.score(X, y))

		# Error de test del modelo 
		# ==============================================================================
		predicciones_modelo1 = modelo.predict(X = X_test)
		print(predicciones_modelo1[0:3,])

		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_modelo1,
			squared = False
		)

		print("")
		print(f"El error (rmse) de test es: {rmse}")
		print("")
		print("")

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)
		# Creación del modelo utilizando matrices como en scikitlearn
		# ==============================================================================
		# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
		X_train = sm.add_constant(X_train, prepend=True)
		modelo = sm.OLS(endog=y_train, exog=X_train,)
		modelo = modelo.fit()
		print(modelo.summary())

		# Intervalos de confianza para los coeficientes del modelo
		# ==============================================================================
		modelo.conf_int(alpha=0.05)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones.head(4)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones['x'] = X_train[:, 1]
		predicciones['y'] = y_train
		predicciones = predicciones.sort_values('x')

		# Error de test del modelo 
		# ==============================================================================
		X_test = sm.add_constant(X_test, prepend=True)
		predicciones_error = modelo.predict(exog = X_test)
		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_error,
			squared = False
			)
		print(f"El error (rmse) de test es: {rmse}")

		# Imprimimos las regresiones
		# ==============================================================================
		print("")
		print("Regresión Linear 1:", f"{self.m1[0][1]}x+{self.b1[0]}")
		print("")
		print("")

		# Gráfico del modelo
		# ==============================================================================
		#fig, ax = plt.subplots(figsize=(6, 3.84))
		
		ax.set_title(f"{title} Linealizada");
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)

		ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
		ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label=f"{self.m1[0][1]}x + {self.b1[0]}")
		ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
		ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
		ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
		ax.legend(fontsize=legendsize);

		plt.savefig(figname)
		plt.show()
	
	# Destroyer function
	def __del__ (self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")

## Graph linear regression only graph without CI's curve
class LSOGComp:
	# Constructor function
	def __init__ (self, xs, ys, title, xlabel, ylabel, figname, legendsize):
		warnings.filterwarnings('ignore')

		print("Calling constructor")
		class_name = self.__class__.__name__
		print(class_name, "Ready")
		
		varindependent = xs
		vardependent = ys
		
		datos = pd.DataFrame({'varindependent': varindependent, 'vardependent': vardependent})
		datos.head(3)
		print("")
		print(datos)
		print("")

		# Gráfico
		# ==============================================================================
		# set params characteristics at all plots and subplots for implements latext
		params = {'text.latex.preamble': [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
		plt.rcParams.update(params)

		# Initialise the figure and a subplot axes. Each subplot sharing (showing) the
		# same range of values for the x and y axis in the plots
		fig, ax = plt.subplots(1, 1,figsize=(6, 6), sharex=True, sharey=True)

		# Set the title for the figure
		#fig.suptitle("Linealización", fontsize=10)

		datos.plot(
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

		# Correlación lineal entre las dos variables
		# ==============================================================================
		corr_test = pearsonr(x = datos['varindependent'], y =  datos['vardependent'])
		print("Coeficiente de correlación de Pearson: ", corr_test[0])
		print("P-value: ", corr_test[1])

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)

		# Creación del modelo
		# ==============================================================================
		modelo = LinearRegression()
		modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

		# Información del modelo
		# ==============================================================================
		print("Intercept (b):", modelo.intercept_)
		self.b1 = modelo.intercept_
		self.m1 = list(zip(X.columns, modelo.coef_.flatten(), ))
		print("Coeficiente (m):", list(zip(X.columns, modelo.coef_.flatten(), )))
		print("Coeficiente de determinación R^2:", modelo.score(X, y))

		# Error de test del modelo 
		# ==============================================================================
		predicciones_modelo1 = modelo.predict(X = X_test)
		print(predicciones_modelo1[0:3,])

		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_modelo1,
			squared = False
		)

		print("")
		print(f"El error (rmse) de test es: {rmse}")
		print("")
		print("")

		# División de los datos en train y test
		# ==============================================================================
		X = datos[['varindependent']]
		y = datos['vardependent']

		X_train, X_test, y_train, y_test = train_test_split(
			X.values.reshape(-1,1),
			y.values.reshape(-1,1),
			train_size   = 0.8,
			random_state = 1234,
			shuffle      = True
		)
		# Creación del modelo utilizando matrices como en scikitlearn
		# ==============================================================================
		# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
		X_train = sm.add_constant(X_train, prepend=True)
		modelo = sm.OLS(endog=y_train, exog=X_train,)
		modelo = modelo.fit()
		print(modelo.summary())

		# Intervalos de confianza para los coeficientes del modelo
		# ==============================================================================
		modelo.conf_int(alpha=0.05)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones.head(4)

		# Predicciones con intervalo de confianza del 95%
		# ==============================================================================
		predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
		predicciones['x'] = X_train[:, 1]
		predicciones['y'] = y_train
		predicciones = predicciones.sort_values('x')

		# Error de test del modelo 
		# ==============================================================================
		X_test = sm.add_constant(X_test, prepend=True)
		predicciones_error = modelo.predict(exog = X_test)
		rmse = mean_squared_error(
			y_true  = y_test,
			y_pred  = predicciones_error,
			squared = False
			)
		print(f"El error (rmse) de test es: {rmse}")

		# Imprimimos las regresiones
		# ==============================================================================
		print("")
		print("Regresión Linear 1:", f"{self.m1[0][1]}x+{self.b1[0]}")
		print("")
		print("")

		# Gráfico del modelo
		# ==============================================================================
		#fig, ax = plt.subplots(figsize=(6, 3.84))
		
		ax.set_title(f"{title} Linealizada");
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)

		ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
		ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label=f"{self.m1[0][1]}x + {self.b1[0]}")
		ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
		ax.legend(fontsize=legendsize)

		plt.savefig(figname)
		plt.show()
	
	# Destroyer function
	def __del__ (self):
		class_name = self.__class__.__name__
		print(class_name, "destroyed")
