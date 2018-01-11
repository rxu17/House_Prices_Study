# Machine Learning: Predicting House Prices

# Software and Libraries
- sklearn
- numpy
- pandas
- statsmodels
- python 3.6
- matplotlib

# Goal
The goal of the project is to predict the sales price for each house using the best variables in Python. 

# Description:
Housing data for houses in Ames, Iowa
There are 79 explanatory variables (included in a data variable description txt file)

Here are some explanatory variables:
- **SalePrice** - the property's sale price in dollars. This is the target variable that you're trying to predict.
- **MSSubClass:** - The building class
- **MSZoning:** - The general zoning classification
- **LotFrontage:** - Linear feet of street connected to property
- **LotArea:** - Lot size in square feet
- **Street:** - Type of road access
- **Alley:** - Type of alley access
- **LotShape:** - General shape of property
- **LandContour:** - Flatness of the property

Others can be found on the [Kaggle site](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

The prediction is evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
 
