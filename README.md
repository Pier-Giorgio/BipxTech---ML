# BIPxTech-TeamSystem: Analysis of Iva data and implement of Machine learning model 
### **1 INTRODUCTION** 
In this project, we aim to develop a predictive model capable of determining the VAT exemption code (IvaM) for each line item in an invoice, leveraging the information from other fields within the same line item. The accurate prediction of VAT exemption codes is crucial for financial accuracy, regulatory compliance, and efficient invoice processing in various business contexts.<br>

Ensuring financial accuracy through correctly predicting VAT exemption codes is crucial for maintaining the integrity of financial records. This accuracy helps prevent costly errors and discrepancies that could arise from incorrect VAT calculations. Moreover, automating the prediction of VAT exemption codes enhances operational efficiency by streamlining the invoicing process. This automation reduces the time and effort required for manual entry and validation, leading to faster processing times, lower administrative costs, and improved overall productivity.<br>

Invoices typically contain a wealth of information spread across multiple fields, such as product descriptions, articles, and various tax-related details. The challenge is to utilize this multi-dimensional data effectively to predict the correct VAT exemption code. Our dataset includes these fields, with the "IvaM" column serving as the target variable for our predictions.<br>

The project involves several key stages, starting from the initial data exploration and pre-processing to the development, evaluation, and implementation of the predictive model. Each stage is critical to ensure the robustness and accuracy of the final model. Below, we outline the main objectives and tasks undertaken:

•**Data exploration and analysis**: the initial phase involves a comprehensive analysis of the dataset to understand its structure, identify key features, and address any anomalies such as missing or unmapped values. This step is essential for setting a solid foundation for subsequent modeling efforts.<br> 
•**Data pre-processing**: this step involves cleaning the dataset, handling missing values, encoding categorical variables, and normalizing numerical features; effective pre-processing is crucial forenhancing the model's performance.<br>
•**Model development**: the core of the project is the development of the predictive model. Machine learning algorithms such as random forests and artificial neural networks are explored so as to find the 
model best suited to our goal. The choice of the final model is based on a thorough evaluation of their performance on the validation dataset.<br>
•**Model evaluation**: the selected model is rigorously evaluated using appropriate metrics such as accuracy and test loss. Comparisons between identical models but made up of different structures are also
used to understand and guarantee the best possible reliability and stability of performance for the chosen model.<br>
•**User interface development**: to facilitate user interaction with the model, a simple and intuitive user interface is developed. This interface allows users to input invoice line item details and receive predicted VAT exemption codes in real-time.<br>

<br>

### **2 METHODS**
We begin our project with the preprocessing of the dataset and the initial Exploratory Data Analysis (EDA) that allow us to have an overview of our data. 

### **2.1 INTRODUCTION TO DATA SCRUB** 
The objective of this section is to prepare the data for analysis and build a preliminary understanding of the dataset through Exploratory Data Analysis (EDA) techniques. Preprocessing and EDA are crucial steps in the machine learning process, as they ensure that the data are clean, consistent, and properly structured for use in predictive models. Preprocessing involves handling missing values, removing duplicates, correcting outliers and transforming variables, while EDA focuses on analyzing distributions, relationships and patterns in the data.

### **2.2 DATASET DESCRIPTION**
The dataset that was presented to us for our Machine Learning project was kindly sent to us by BipxTech for the creation of an exemption code mapper; it is initially composed of 134437 observations, distributed over 45 different variables, so it represents a very full-bodied dataset of information and data, so it required on our part a thorough analysis of the variables both from an economic point of view, to understand in depth what we were talking about, as well as of course from a statistical point of view; this is intended for developing a machine learning model to predict exemption VAT codes. Each record in the dataset represents an invoice line, containing various fields that provide detailed information about the transaction.
Additional fields provide further context and details necessary for understanding the transactions and identifying patterns relevant to predicting the exemption VAT code. Data cleaning and preprocessing will be required to handle any missing or inconsistent entries, ensuring the dataset is ready for model training and analysis.

### **2.3 DATA COLLECTION**
As a first step in our project we took care to import all the libraries we need to perform our analysis; therefore, we used some essential libraries for data manipulation, visualization, preprocessing, machine learning, and neural networks. The Pandas library is very important for data manipulation and analysis, while openpyxl is used to read and write Excel files. Seaborn is a high-level interface for creating attractive statistical graphs, while Matplotlib allows us to create static, animated, and interactive visualizations. NumPy supports high-performance operations on arrays and matrices and contains an extensive library of mathematical functions for scientific computing. Finally, Scikit-learn is a leading machine learning library and offers tools for data preprocessing and model building.

-# We import all the libraries that we need for our analysis<br>
_import pandas as pd<br>
!pip install openpyxl #We need this library to read the file in xlsx format<br>
import seaborn as sns<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
from sklearn.preprocessing import OrdinalEncoder<br>
from sklearn.preprocessing import MinMaxScaler<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.preprocessing import LabelEncoder<br>
from tensorflow.keras.models import Sequential<br>
from tensorflow.keras.layers import Dense<br>
from tensorflow.keras.utils import to_categorical_<br>

So here we have loaded our dataset, loaded our path, and created our dataframe (df). 
Our preliminary analysis then begins with the functions of df.shape that allowed us to visualize the dimension of the dataset, will display the number of observations (rows) and
features (columns) in the dataset then continuing with the display of the first 10 rows of our dataset and the last 10, for visualize the first and last observations.<br>

-# Load the path to read the dataset<br>
_bip_data ='/content/luiss_data_anonym.xlsx' # Everyone enter their own path to file<br>
df = pd.read_excel(bip_data, engine='openpyxl')<br>
print(df)_<br>

_df.shape_<br>

_df.head(10)_<br>

_df.tail(10)_<br>

We continued with the command (.info()) helps us understand the data type and information about our data, including the number of records in each column, the data that has null or non-null, the data type and memory usage of the dataset.<br>

_df.info()_<br>

We took a look at duplicate checking: based on several unique values in each column and data description, we can identify continuous and categorical columns in the data; duplicate data can be handled or removed based on further analysis.<br>

_df.nunique()_<br>

We study again the type of variables that are present in the dataset through the command (df.dtypes), displaying the type of variables clearly; we then closed this initial part of data collection and initial data visualization with the command (df.count()), which allowed us to get a general idea of the number of observations in each column, allowing us to also understand the number of missing values present within our dataset.<br>

-# We study again the type of variables that are present in the dataset<br>
_print(df.dtypes)_<br>

-# We calculate items count for each column<br>
_df.count()_<br>

### **2.4 DATA PROCESSING: DATA SCRUB** 
The preprocessing phase involves understanding, visualizing, and analyzing data to gain insights into its characteristics and relationships. This process includes cleaning, transforming, and organizing raw data into a suitable format for the next stages of analysis. The specific methods and techniques used depend on the data's nature, the machine learning (ML) algorithms employed, and the analysis objectives. Proper data analysis and preprocessing are crucial for enhancing the quality and effectiveness of ML models. Our goal is to improve our Machine Learning model's performance through thorough preprocessing steps.<br>
Preprocessing Steps:<br>
1. **Understanding Data**
   - Comprehending the structure, types, and general characteristics of the dataset.<br>
   - Identifying the variables and their roles in the analysis.<br>
2. **Handling Missing Values**
   - Detecting and managing missing or null values within the dataset.<br>
   - Techniques include imputation or removal of missing data.<br>
3. **Variable Identification**
   - Categorizing variables as numerical or categorical.<br>
   - Understanding the nature and distribution of each variable.<br>
4. **Checking for Duplicate Rows**
   - Identifying and removing duplicate entries to ensure data integrity.<br>
5. **Dealing with Outliers**
   - Detecting and handling outliers that may skew the analysis.<br>
   - Strategies include capping, flooring, or removal of outlier values.<br>
6. **Visualization and Analysis**
   - Creating visualizations to understand data distributions and relationships.<br>
   - Tools used include Matplotlib and Seaborn for plotting.<br>
7. **Normalizing Numerical Variables**
   - Scaling numerical features to a standard range, typically 0 to 1.<br>
   - Methods include Min-Max Scaling or Standardization.<br>
8. **Encoding Categorical Variables**
   - Converting categorical data into numerical format for ML algorithms.<br>
   - Techniques include One-Hot Encoding and Label Encoding.<br>
9. **Feature Selection or Dimensionality Reduction**
   - Selecting relevant features or reducing dimensionality to improve model performance.<br>
   
We then plunged into the preprocessing of our data moving immediately to step two which is about handling missing values, since either way we had already initially studied our dataset as described above, and either way step number 1 is : understand the data, comprehending the structure, types, and general characteristics of the dataset, identifying the variables and their roles in the analysis.<br>

### **2.4.1 HANDLING MISSING VALUES**
So to handle our missing values, we immediately displayed through the function (df.isnull().sum()) the total number of each Nan values present in each row of our dataframe and to have an overview on the presents of this missing values <br>

-# This step is crucial to identify null values in our dataframe<br>
_df.isnull().sum()_ <br>

We then found that our dataset was reporting a massive amount for many rows, a little less than half, and so we continued in our study of the missing values, trying through better data visualizations, to figure out in what amounts they were missing for each row; and in the meantime we also decided to drop, delete the initial column 'Unnamed: 0' which represented only a circumstance variable, counting only the number of rows in the dataset and therefore completely useless for the purposes of our analysis.<br>

-# We drop the first variable that are not useful for our predictive aim
_df = df.drop(['Unnamed: 0'], axis = 1)<br>
df.info()_ <br>

Returning to the display of the missing values through this command, we were able to translate into percentages, the presence of missing values for each column, so that we had a clear idea of the columns that we necessarily had to delete.<br>
 
-# The below code helps us to calculate the percentage of missing values in each column
_(df.isnull().sum()/(len(df)))*100_ <br>

As you can see, there are many variables that contain many Nan, some even close to or equal to 100% of the total.<br>
We decided to eliminate the columns that have a percentage of missing values greater than 50%, to avoid that our analysis became misleading, allowing us to facilitate the in-depth understanding of the dataset, since these columns are so empty it was also difficult to impute them through an average value or a median, risking to make mistakes; the columns in question are:<br>
-	**C** : % pro rata of activity<br>
-	**E** : Type of withholding<br>
-	**F**: Reason for withholding<br>
-	**G** : Type of second document withholding<br>
-	**H** : Reason for second withholding of document<br>
-	**CE** : Type of transfer (AB, PR, etc.). CO identifies the contribution lines.<br>
-	**Comp** : VAT offset for sales made by companies under special agricultural regime.<br>
-	**Iva11** : VAT regime applied.<br>
-	**%Forf** : flat rate %.<br>
-	**Nomenclature** : Internal nomenclature. Retrieved for companies with item mapping and retrieved from the management system's item master.<br>
-	**Ritac** : Standard withholding table code<br>
-	**RF** : Electronic invoice tax regime.<br>
-	**RifNormative** : Regulatory reference<br>
-	**Rifamm** : Administrative reference in the electronic invoice<br>
-	**Art2** : Second article present in the electronic invoice<br>
-	**Value2** : Value of the second item present in the electronic invoice<br>
-	**Art3** : Third item present in the electronic invoice<br>
-	**Value3** : Value of the third item present in the electronic invoice<br>

So firstly, we recalculate the percentage of missing values for each column, and next, we identify the columns that have more than 50% missing values; we then remove these columns from the original DataFrame, and lastly, we display the remaining columns to confirm the operation. This process is useful for cleaning the data, as it eliminates columns that contain too many missing values, which could compromise the quality of the analysis or the implementation of Machine Learning model.<br>

-# So we recalculate the percentage of missing values for each column as above<br>
_missing_percentage = df.isnull().mean() * 100_ <br>

-# We identifies the columns with more than 50% missing values<br>
_columns_to_drop = missing_percentage[missing_percentage > 50].index_ <br>

-# We delete these columns directly from the original DataFrame<br>
_df.drop(columns=columns_to_drop, inplace=True)_ <br>

-# Display the remaining columns to confirm the operation<br>
_print(df.columns)_ <br>
 
-# So these are the column remaining in our analysis<br>

As can be seen, we now have a smaller number of columns (variables), 19 columns were eliminated after the missing value analysis and 26 remain to continue our analysis.<br>

Now we need to handle the columns that have Nan less than 50%, but are still there even if in a small percentage. They are there, even if in a small percentage. So we decide to recalculate the percentage of missing values for the remaining columns. <br>

_(df.isnull().sum()/(len(df)))*100_ <br>

To allow us to have a better visualization we decide to make a plot, through the missingno library, which makes us visualize Nan in real time all together.<br>

-# We plot the remaining missing values to visualize them in graphics <br>
_import missingno as msno_ <br> 
_msno.matrix(df)_ <br>

-# This graph below allows us to see the remaining Nan all together, in fact as we can see there are no more dropped columns, and allow us to visualize them in an easier way; <br>
![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/229322d4-26f5-4bc8-a7be-ba55a4b5c44f)<br>
Doing a count of the remaining missing values, in order to have accurate numbers, we realized that however both the number of the latter considered all together was still nonetheless both important, precisely 8163 Nan that still need to be handled to improve our dataset. <br>

Then we display the variable types that have a small percentage of Nan; we can see an interesting thing, that all these columns with Nan are categorical variables (object type) and therefore difficult to handle from the point of view of missing values; so the best solution for us is to delete these rows with missing values or replace these values with the observation “Not known” or “Missing value.” the columns in question are:<br>
-	**B** : Business with deferred VAT <br>
-	**D** : VAT exigibility of the document<br>
-	**DescrizioneRiga**: description of the row (first 98 characters)<br>
-	**IvaM** : Exemption codes. 
-	**Art1** : Article derived from the XML invoice. The field accepts the data field type.<br>
-	**Value1** : Article code derived from the XML invoice. The field accepts the value field of the XML trace<br>
-	**CMar** : Margin management type value on the accounting reason.<br>
-	**CTra** : Autotransport flag value on the accounting reason.<br>
-	**Rev** : Reverse charge reason <br>
-	**X** : Subject to pro rata for the period prior to the invoice document date<br>

In this code segment, first, we decided to remove all rows that contained missing values, reassigning the resulting DataFrame to the df variable. Next, we verified the effectiveness of this cleanup by printing the number of missing values for each column, confirming that there were no more missing values. Finally, we saved the cleaned dataset in a CSV file called cleaned_dataset.csv. These steps were crucial in ensuring that our dataset was free of missing values, thus clearly improving the quality of our analysis. <br>

-# So we decide to drop the rows that present Nan values; we drop rows with missing values and reassign to df
_df = df.dropna()_ <br>

-# Display the number of missing values per column after cleaning<br>
_print("Missing values per column after cleaning:")_ <br>
_print(df.isnull().sum())_ <br>

-# Save the cleaned dataset <br>
_df.to_csv('cleaned_dataset.csv', index=False)_ <br>

We now have no more missing values, but we also have fewer rows of total observations, having removed all those that had at least one Nan. We now have 126,471 observations, as can be seen from the count. Thus ends our first part of Nan imputation. <br>

### **2.4.2 CHECKING DUPLICATE ROWS**
Now we move on to checking for duplicates, although in any case we have to be careful at this stage and choose the best strategy for conducting our analysis; we are not going to delete all duplicates, because our analysis however it is based on an accounting type dataset, each billing row can reveal important information to us; for this reason we decided to delete only the rows that are entirely the same, dropping the second one and keeping the first one.<br>

-# We try to check for fully duplicate rows<br>
_duplicate_rows = df[df.duplicated(keep=False)]_ <br>
_print("Number of fully duplicate rows:", duplicate_rows.shape[0])_ <br>
-# And print fully duplicate rows
_print(duplicate_rows)_ <br>
-# Remove fully duplicate rows, keeping the first
_df = df.drop_duplicates(keep='first')_ <br>
-# Save the cleaned dataset <br>
_df.to_csv('cleaned_dataset.csv', index=False)_ <br>

We complete this part by counting the remaining rows after removing duplicate rows and completing the preprocessing and cleaning of our data set, resulting in 114454 rows and 26 columns, which we can use to continue the study and subsequent implementation of our model. <br>

### **2.5 EDA: EXPLORING FEATURES OF THE DATASET**
Exploratory Data Analysis refers to the crucial process of performing initial investigations on data to discover patterns to check assumptions with the help of summary statistics and graphical representations.<br>  
-	EDA can be leveraged to check for outliers, patterns, and trends in the given data.<br>
-	EDA helps to find meaningful patterns in data.<br>
-	EDA provides in-depth insights into the data sets to solve our business problems.<br>

As a reminder of what kind of numerical variables we are talking about here, I am going to list below all specific meaning for each of these So the columns in question are:<br>
-	**Ateco** : Ateco code<br>
-	**DataDoc** : Document date<br>
-	**Importo** : Amount of the row<br>
-	**Conto** : Account resulting from mapping<br>
-	**ContoStd** : Related standard account<br>
-	**IvaM** : Exemption codes (table to follow)
-	**TM** : Mapping type (table to follow)<br>
-	**%RIT1** : % first advance withholding<br>
-	**%RIT2** : % second advance withholding<br>
-	**CoDitta** : Company code ts-studio<br>
-	**TIva** : VAT type on the reason<br>
-	**Caus** : Possible standard causal code (VAT type if the standard causal code is absent).<br>

By using the df.describe() function we get an idea of the statistical characteristics of our numerical variables so we can understand any outliers in our variables, and also it gives us information from the point of view of the patterns and features that are there.<br>

-# Provide a statistics summary of data belonging to numerical datatype such as int, float<br>
_df.describe().T_ <br>

Before proceeding with EDA, we separate numerical and categorical variables to facilitate analysis.<br>

_cat_cols=df.select_dtypes(include=['object']).columns_ <br>
_num_cols = df.select_dtypes(include=np.number).columns.tolist()_ <br>
_print("Categorical Variables:")_ <br>
_print(cat_cols)_ <br>
_print("Numerical Variables:")_ <br>
_print(num_cols)_ <br>

### **2.5.1 CHECK AND MANAGE OUTLIERS**
In this section we deal with the identification and management of outliers. Outliers can have a significant impact on the analysis and results. However, since this is an economic project and we cannot verify the correctness of every observation, we prefer to keep outliers without erroneously modifying the dataset. The only outliers are specific variables that we can safely modify to stay within defined ranges. <br>
Through this code we tried to visualize through Box plots the distribution of our numerical variables to understand which and how our variables were subject to the presence of the latter. <br>

-# We have to identify numerical columns <br>
_num_cols = df.select_dtypes(include=np.number).columns.tolist()_  <br>
_print("Numerical Variables:", num_cols)_  <br>

-# Going on we visualize outliers using box plots<br>
_for column in num_cols:_ <br>
_plt.figure(figsize=(10, 6))_ <br>
_sns.boxplot(x=df[column])_ <br>
_plt.title(f'Box plot of {column}')_ <br>
_plt.show()_ <br>

-# And detect outliers using the IQR method<br>
_outliers = pd.DataFrame()_ <br>

_for column in num_cols:_ <br>
_Q1 = df[column].quantile(0.25)_ <br>
_Q3 = df[column].quantile(0.75)_ <br>
_IQR = Q3 - Q1_ <br>
_lower_bound = Q1 - 1.5 * IQR_ <br>
_upper_bound = Q3 + 1.5 * IQR_ <br> 
_outliers_in_column = df[(df[column] < lower_bound) | (df[column] > upper_bound)]_ <br>
_outliers = pd.concat([outliers, outliers_in_column])_  <br>

_outliers = outliers.drop_duplicates()_ <br>
_print(f"Number of outliers detected: {outliers.shape[0]}")_ <br>

-# Finally we print and inspect the outliers<br>
_print(outliers)_ <br>

We then obtain from this graphical visualization, an overview of the presence of the various outliers in our numerical columns; as we expounded earlier, we decided to ''trust'' the data and not eliminate outliers for variables that do not have values within certain ranges. The only exception therefore are the two variables Ivam and TM, which are defined within certain ranges of values.<br>
For example, for IvaM we know that the values range from 300 to 382, so values above and below this range are definitely invalid, so the only variables that have predetermined limits for which outliers can be misleading are only IvaM and TM, but TM having only values within its range (2-20) does not create any problem for us, the only one that needs to be changed is IvaM.<br>

In this code, we defined a valid range for the `IvaM` column in our dataset. The range is from a minimum of 300 to a maximum of 382. Next, we filtered the rows of the dataset to keep only those where the `IvaM` value falls within this range. The resulting dataset was then saved to a CSV file named `cleaned_dataset.csv`. Finally, we displayed the shape of the new DataFrame and the first few rows to verify the cleaning process. This process allowed us to retain only the data that falls within the specified range, thereby improving the overall quality of our dataset. <br>
 
-# We define the valid range for IvaM<br>
_valid_range_min = 300_ <br>
_valid_range_max = 382_ <br>

-# And filter out the rows where IvaM is outside the valid range <br>
_df_filtered = df[(df['IvaM'] >= valid_range_min) & (df['IvaM'] <= valid_range_max)]_ <br>

-# We save the cleaned dataset to CSV file_<br>
_df_filtered.to_csv('cleaned_dataset.csv', index=False)_ <br>

-# Display the shape of the new dataframe and first few rows <br>
_print(df_filtered.shape)_ <br>
_df_filtered.head()_ <br>

We display again the summary of statistical features to see how it changed for IvaM and now we can consider it correct. <br>

_df_filtered.describe().T_ <br>

### **2.5.2 UNIVARIATE ANALYSIS**
Univariate analysis involves analyzing and visualizing the dataset by examining one variable at a time. This type of analysis is fundamental for understanding the individual characteristics of each variable, which helps in identifying patterns, distributions, and potential outliers. Effective data visualization is crucial in this process, as it allows us to gain insights into the data and make informed decisions about subsequent analyses.<br>
When performing univariate analysis, it is important to choose the appropriate type of chart or plot based on the nature of the variable. Univariate analysis can be conducted on both categorical and numerical variables, each requiring different visualization techniques.<br>

Categorical variables represent discrete categories or groups. To visualize these variables, we can use the following plots:<br>
- **Count Plot**: A count plot displays the frequency of each category in the variable. It is useful for understanding the distribution and prevalence of different categories.<br>
- **Bar Chart**: Similar to a count plot, a bar chart represents the frequency of categories using bars. It is an effective way to compare the sizes of different groups.<br>
- **Pie Plot**: A pie plot shows the proportion of each category as segments of a circle. This plot is useful for visualizing the relative sizes of categories in a variable.<br>

Numerical variables represent continuous or discrete numerical values. For these variables, we can use the following plots:<br>
- **Histogram**: A histogram displays the distribution of a numerical variable by dividing the data into bins and plotting the frequency of values in each bin. It helps in understanding the shape, central tendency, and spread of the data.<br>
- **Box Plot**: A box plot shows the distribution of a numerical variable through its quartiles, highlighting the median, interquartile range, and potential outliers. It is useful for comparing distributions and identifying outliers.<br>
- **Density Plot**: A density plot is a smoothed version of a histogram, representing the distribution of a numerical variable as a continuous probability density curve. It provides a clearer view of the distribution's shape and variability.<br>

In our example, we conducted a univariate analysis on continuous variables using histograms and box plots. The histogram allowed us to visualize the distribution of the data, revealing the frequency and spread of values. The box plot complemented this by providing a summary of the data's central tendency, dispersion, and potential outliers. <br>

By performing univariate analysis, we gained valuable insights into the individual characteristics of each variable. This preliminary step is essential for identifying patterns and anomalies in the data, guiding further exploration and analysis in our machine learning project. <br>

So in this section, we used histograms and box plots to visualize the behavior of the numerical variables in our dataset. This approach helps in identifying patterns, skewness, and outliers in the data. Here's a detailed explanation of the process: <br>

1. **Calculating Skewness**:<br>
   For each numerical variable, we calculated the skewness, which measures the asymmetry of the distribution of values. A skewness value close to zero indicates a symmetric distribution, while positive or negative skewness indicates a distribution that is skewed to the right or left, respectively.<br>

2. **Visualizing the Variables**:<br>
   To better understand the distribution of the numerical variables, we created two types of plots for each variable:<br>
   
   - Histogram: This plot shows the frequency of values of the variable divided into intervals (bins). It is useful for visualizing the overall shape of the distribution, identifying any skewness, and observing the density of the data.<br>
   
   - Box Plot: This plot summarizes the distribution of the variable by showing the quartiles, median, and potential outliers. It is particularly useful for identifying the spread of the data and the presence of outliers.<br>

3. **Execution of the Code**:<br>
  For each numerical variable in our dataset:<br>
   - We printed the name of the variable and its skewness value rounded to two decimal places.<br>
   - We created a figure with two side-by-side subplots. In the first subplot, we plotted the histogram of the variable to show its distribution. In the second subplot, we plotted the box plot to highlight the median, quartiles, and outliers.<br>
   - Finally, we displayed the figure to visualize the results.<br>

Here's a step-by-step explanation of the code:<br>

- The `for` loop iterates through each numerical column in the dataset.<br>
- For each column, the skewness of the variable is calculated and printed.<br>
- A figure with two side-by-side subplots is created.<br>
- In the first subplot, a histogram of the variable is plotted without grid lines to display the distribution of values.<br>
- In the second subplot, a box plot of the variable is plotted to highlight the median, quartiles, and outliers.<br>
- The figure is then displayed to show the visualizations.<br>

This approach allowed us to gain a clear and detailed view of the distribution of the numerical variables, making it easier to identify skewness and outliers that could affect subsequent analyses or the performance of machine learning models.<br>

-# In the below figure, a histogram and box plot is used to show the pattern of the variables, as some variables have skewness and outliers. <br>
_for col in num_cols:_ <br>
_print(col)_ <br>
_print('Skew :', round(df_filtered[col].skew(), 2))_ <br>
_plt.figure(figsize = (15, 4))_ <br>
_plt.subplot(1, 2, 1)_ <br>
_df[col].hist(grid=False)_ <br>
_plt.ylabel('count')_ <br>
_plt.subplot(1, 2, 2)_ <br> 
_sns.boxplot(x=df_filtered[col])_ <br> 
_plt.show()_ <br>

In this another section, we visualized the distribution of the first set of categorical variables in our dataset using bar plots. This approach helps us understand how frequently each category appears within these variables.<br>

First, we set up a figure with multiple subplots, creating a grid of eight plots arranged in four rows and two columns. This layout allowed us to display the bar plots for multiple variables simultaneously. We also added a title to the figure to describe its content. <br> 

Next, we configured each subplot to ensure the labels on the x-axis are readable. By rotating the labels by 90 degrees and adjusting their size, we made sure that even long category names are clearly visible.
For each categorical variable, we created a bar plot that shows the frequency of each category. We ordered the categories by their frequency, displaying the most common categories first. Here’s what we did for each variable:<br> 
●	For the first variable, we created a bar plot showing how often each category appears. <br> 
●	We repeated this process for the second variable, the third variable, and so on, up to the eighth variable. <br> 
●	For some variables with a large number of categories, we limited the plot to show only the top 20 most frequent categories. <br> 
Finally, we displayed all the plots together in the figure. This visual representation allowed us to easily see the distribution of categories within each variable, providing valuable insights into the structure of our data.. <br> 
Below we then analyzed the first set of categorical variables: <br> 
-	**A** : Business type <br> 
-	**B** : Business with deferred VAT <br> 
-	**D** : VAT exigibility of the document <br>  
-	**Tdoc : Document Type <br>  
-	**VA** : Document type sales (V) or purchases (A) <br>  
-	**DescrizioneRiga : Description of the row (first 98 characters) <br>  
-	**Iva** : Nature or VAT rate applied <br>  
-	**Art1** : Article deriving from the XML invoice. The field accepts the data type field <br>

-# We create bar plots for the first set of categorical variables <br>
_fig, axes = plt.subplots(4,2, figsize = (20, 20))_ <br> 
_fig.suptitle('Bar plot for the first set of categorical variables in the dataset', fontsize=15)_ <br>
_for row in axes:_ <br>
_for ax in row:_ <br> 
_ax.tick_params(labelrotation=90, labelsize=10)_ <br> 
_sns.countplot(ax = axes[0, 0], x = 'A', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['A'].value_counts().index);_ <br> 
_sns.countplot(ax = axes[0, 1], x = 'B', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['B'].value_counts().index);_ <br> 
_sns.countplot(ax = axes[1, 0], x = 'D', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['D'].value_counts().index);_ <br> 
_sns.countplot(ax = axes[1, 1], x = 'Tdoc', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['Tdoc'].value_counts().index);_ <br> 
_sns.countplot(ax = axes[2, 0], x = 'VA', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['VA'].head(20).value_counts().index);_ <br> 
_sns.countplot(ax = axes[2, 1], x = 'DescrizioneRiga', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['DescrizioneRiga'].head(20).value_counts().index);_ <br> 
_sns.countplot(ax = axes[3, 0], x = 'Iva', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['Iva'].value_counts().index);_ <br> 
_sns.countplot(ax = axes[3, 1], x = 'Art1', data = df_filtered, color = 'blue',_ <br> 
_order = df_filtered['Art1'].value_counts().index);_ <br> 
_plt.show()_ <br> 

And below we do the same thing with the second set of categorical variables:<br> 
-	**CMar** : Margin management type value on the accounting reason <br> 
-	**CTra** : Autotransport flag value on the accounting reason <br> 
-	**Rev**: Reverse charge reason <br>  
-	**CVia** : Travel agency reason <br> 
-	**X** : Subject to pro rata for the period prior to the invoice document date <br>

-# We creating a subplot structure <br>
_fig, axes = plt.subplots(3, 2, figsize=(20, 20))_ <br>
_fig.suptitle('Bar plot for the second set of categorical variables in the dataset', fontsize=30)_ <br>

-# And setting up individual plots <br>
_sns.countplot(ax=axes[0, 1], x='CMar', data=df_filtered, color='blue', order=df_filtered['CMar'].value_counts().index)_ <br>
_sns.countplot(ax=axes[1, 0], x='CTra', data=df_filtered, color='blue', order=df_filtered['CTra'].value_counts().index)_ <br>
_sns.countplot(ax=axes[1, 1], x='Rev', data=df_filtered, color='blue', order=df_filtered['Rev'].value_counts().index)_ <br>
_sns.countplot(ax=axes[2, 0], x='CVia', data=df_filtered, color='blue', order=df_filtered['CVia'].head(20).value_counts().index)_ <br>
_sns.countplot(ax=axes[2, 1], x='X', data=df_filtered, color='blue', order=df_filtered['X'].head(20).value_counts().index)_ <br>

-# Finally adjusting tick parameters after plotting to ensure they apply to all subplots <br>
_for row in axes:_ <br>
_for ax in row:_ <br>
_ax.tick_params(labelrotation=90, labelsize=10)_ <br>

_plt.show()_ <br>

In this other part of the analysis, we focused on the numerical columns of our dataset to understand the relationships between them. <br>
First, we selected only the numerical columns from our dataset. These are the columns that contain numerical data, such as integers or floating-point numbers. By filtering out the non-numerical columns, we created a new dataset that includes only the numerical information. <br>

Next, we calculated the correlation matrix for these numerical columns. A correlation matrix is a table that shows the correlation coefficients between pairs of variables. The correlation coefficient is a measure of how strongly two variables are related to each other. Values close to 1 or -1 indicate strong relationships, while values close to 0 indicate weak or no relationships.  <br>

After calculating the correlation matrix, we displayed it. This matrix helps us identify which numerical variables are closely related. For example, if two variables have a high positive correlation, it means that as one variable increases, the other tends to increase as well. Conversely, a high negative correlation indicates that as one variable increases, the other tends to decrease.  <br>

Displaying the correlation matrix provides a clear overview of the relationships between all numerical variables in our dataset. This information is crucial for understanding the structure of the data and can guide further analysis and modeling. Identifying strong correlations can help in feature selection, where we choose the most relevant variables for building our machine learning models.  <br>

-# We select only the numerical columns <br>
_numerical_columns = df_filtered.select_dtypes(include=['float64', 'int64']).columns_ <br>

-# And we calculate the correlation between IvaM and other numerical variables <br>
_correlation_with_IvaM = df_filtered[numerical_columns].corr()['IvaM'].drop('IvaM')_ <br>

-# Display the correlation with IvaM <br>
_print("Correlation of IvaM with other numerical variables:")_ <br>
_print(correlation_with_IvaM)_ <br>

-# Optionally, visualize the correlation with a bar plot <br>
_plt.figure(figsize=(10, 6))_ <br>
_sns.barplot(x=correlation_with_IvaM.index, y=correlation_with_IvaM.values, palette='coolwarm')_ <br>
_plt.title('Correlation of IvaM with Other Numerical Variables')_ <br>
_plt.xlabel('Variables')_ <br>
_plt.ylabel('Correlation Coefficient')_ <br>
_plt.xticks(rotation=90)_ <br>
_plt.show()_ <br>

The graph presented is a bar plot that displays the correlation coefficients between the variable `IvaM` and other numerical variables in the dataset. The correlation coefficient is a statistical measure that indicates the strength and direction of the relationship between two variables.  <br>
In this graph, we can observe how each variable is related to `IvaM`. For instance, we notice that `Conto` and `ContoStd` have a positive correlation with `IvaM`, suggesting that as these variables increase, the value of `IvaM` tends to increase as well. This positive relationship can help us understand which factors might be contributing to higher values of `IvaM`. <br>

On the other hand, variables like `TIva`, `Caus`, and `CoDitta` show a negative correlation with `IvaM`. This means that as these variables increase, the value of `IvaM` tends to decrease. Such negative relationships can indicate factors that are inversely related to `IvaM`, providing a different perspective on how these variables interact.<br>

There are also variables, such as `Importo` and `Ateco`, which have correlation coefficients close to zero. This indicates that there is a weak or no linear relationship between these variables and `IvaM`, suggesting that changes in these variables do not significantly affect `IvaM`.<br>
Understanding these correlations is crucial for several reasons. Firstly, it aids in feature selection by identifying which variables are most relevant for predicting `IvaM`. This can enhance the efficiency and accuracy of predictive models. Secondly, the insights gained from these relationships help us make more informed decisions based on the data. Lastly, recognizing strong correlations can also assist in detecting anomalies or unexpected patterns in the dataset, which could be critical for further analysis or decision-making processes.<br>
Overall, this bar plot provides a clear and concise summary of how `IvaM` is associated with other  numerical variables, offering valuable insights into the underlying structure of the data.<br>
![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/d7abc565-d978-45d1-be90-e85fadb675f3)<br>

### **2.5.3 MULTIVARIATE ANALYSIS**
Multivariate analysis, as the name suggests, involves the examination of more than two variables simultaneously. This type of analysis is incredibly valuable for understanding the complex relationships and interactions between different variables within a dataset. By considering multiple variables at once, multivariate analysis allows us to uncover patterns and insights that would not be apparent when looking at variables in isolation.<br>

One of the most commonly used tools for conducting multivariate analysis is the heat map. A heat map provides a visual representation of the relationships between variables, specifically highlighting the strength and direction of correlations. In a heat map, each cell represents the correlation coefficient between two variables. The color of the cell indicates the strength and direction of the correlation, with different colors representing positive and negative correlations.<br>

By using a heat map, we can easily identify which variables are closely related. This is particularly useful in many applications, such as feature selection for machine learning models, where we want to choose variables that are highly informative about the outcome of interest. Moreover, understanding the correlations between variables helps in identifying multicollinearity, which can be problematic in regression analyses and other statistical models.<br>

Overall, multivariate analysis, facilitated by tools like heat maps, is a powerful method for exploring and understanding the intricate web of relationships within a dataset. It provides a comprehensive view that is essential for making informed decisions and deriving meaningful conclusions from complex data.<br>

In this part of the analysis, we focused on the numerical columns within our filtered dataset. First, we identified and selected these numerical columns, which included data types such as float and integer. By isolating these columns, we created a subset of the dataset that only contained numerical values.<br>
Next, we calculated the correlation matrix for these numerical columns. The correlation matrix is a table that displays the correlation coefficients between pairs of variables. These coefficients measure the strength and direction of the linear relationship between two variables, with values ranging from -1 to 1. A coefficient close to 1 indicates a strong positive correlation, a value close to -1 indicates a strong negative correlation, and values around 0 suggest little to no linear relationship. To visualize the correlation matrix, we created a heat map. This graphical representation uses color to indicate the correlation coefficients between variables. We chose a color scheme ('coolwarm') that effectively highlights the variations in correlation strength and direction. In the heat map, each cell corresponds to a pair of variables, with the color intensity representing the magnitude of their correlation.<br>

The heat map also includes annotations to display the exact correlation coefficients, formatted to two decimal places. This provides a clear and precise view of the relationships between variables. Additionally, we included a color bar to help interpret the color gradients, making it easier to understand the levels of correlation visually.<br>
Finally, we titled the heat map 'Heat Map of Correlation Matrix' to succinctly describe the content of the visualization. Displaying this heat map allowed us to quickly and effectively identify which numerical variables in our dataset were strongly correlated, either positively or negatively. This insight is crucial for understanding the data structure and for guiding further analysis, such as identifying which variables may be redundant due to high correlation or which variables could be influential for predictive modeling.<br>

-# Select only the numerical columns<br>
_numerical_columns = df_filtered.select_dtypes(include=['float64', 'int64']).columns_ <br>
_df_numerical = df_filtered[numerical_columns]_ <br>

-# Calculate the correlation matrix<br> <br>
_correlation_matrix = df_numerical.corr()_ <br>

-# Create a heat map<br>
_plt.figure(figsize=(14, 12))_ <br>
_sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)_ <br>
_plt.title('Heat Map of Correlation Matrix')_ <br>
_plt.show()_ <br>

The heat map we generated represents the correlation matrix for the numerical variables in our dataset. Each cell in this matrix shows the correlation coefficient between two variables, with values ranging from -1 to 1. <br>
From our matrix, we can see that: <br>
-	Conto and ContoStd have a perfect positive correlation (correlation coefficient of 1), indicating they are essentially the same variable. <br>
-	TIva has a moderate positive correlation with Caus, as indicated by the lighter red color with a coefficient of 0.40. <br>
-	Most other variable pairs show weak or no significant correlation, as indicated by the darker blue and light-colored cells. <br>
This visualization helps us quickly identify which variables are closely related and can guide us in making decisions for further analysis, such as feature selection for modeling. <br>
![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/a8f8d425-8902-495b-8542-2812ce2bc4e1) <br>

### **2.6 ENCODING AND NORMALIZATION**
In this section we will describe the process of encoding and normalizing the data to prepare it for training the predictive model. These steps are critical to ensuring that the model can effectively learn from available features.<br>

We have chosen to use Ordinal Encoding to convert the categorical variables to numeric in our dataset. This technique assigns a unique integer value to each category, preserving the natural order of the categories (when it exists). The choice of Ordinal Encoding is primarily motivated by its simplicity and efficiency, as it is easier to implement and occupies less memory than one-hot encoding, which is ideal for working with large datasets. In addition, Ordinal Encoding is particularly compatible with certain machine learning algorithms that can benefit from the use of ordinal variables where the order of the categories could have an important meaning.<br>

To do this, we selected all non-numeric columns in the dataset, then to ensure compatibility with the encoder, all non-numeric columns were converted into strings and finally the categorical columns were transformed into ordinal values using the ordinal encoder.<br>

Concerning the normalisation process (normaliser Min-Max), we chose to scale the numerical variables in a range between 0 and 1. This process was crucial to ensure that all features have equal importance during the training of the model, preventing certain features from dominating due to their different scales. We chose to normalise rather than standardise because the distributions of our data did not follow a normal distribution. This step is particularly useful for models that are sensitive to feature scales, such as models based on neural networks, and can improve the speed and stability of convergence.

<br>

### **3 CHOICE/IMPLEMENTATION OF THE MODEL AND USER INTERFACE**

### **3.1 MODEL IMPLEMENTATION - ANN with 3 layers (256-128-64), 100 epochs, batch size 64**
**Choice of Algorithm: Artificial Neural Network (ANN)**<br>
The aim of our work is to develop a predictive model capable of predicting the VAT exemption code for each invoice line. To do this, we opted for a model based on an artificial neural network (ANN).
The decision to use an artificial neural network was guided by the complex and multidimensional nature of the dataset. Neural networks are particularly effective in recognising and modelling complex, non-linear relationships between input variables, making them ideal for this purpose and are particularly suitable for tabular data and can effectively handle multiclass classification. Furthermore, ANNs' ability to work with a large number of inputs and their flexibility in learning complex patterns make them well suited to make the most of the 26 available variables, which may contain significant hidden patterns for VAT code prediction.

**Use of All Variables**<br>
We initially decided to create an initial model using an 'inclusive' approach, using all 26 variables remaining after the data cleaning process. This decision was based on the project's recommendation to explore possible correlations or patterns between various attributes. Using all available variables makes it possible not to overlook potential important predictors, especially at an early stage of modelling, where the focus is on a broad understanding of the dynamics of the dataset.

**Description of the  model (ANN with 3 layers (256-128-64), 100 epochs, batch size 64)** <br>
The initial phase consisted of loading and preparing the dataset, selecting the independent variables (X) by deleting the column 'VatM', which represents the target for classification. The target (y) was transformed numerically through the use of LabelEncoder and then converted into a categorical format, suitable for neural network output.<br>

For the training phase, we divided the dataset into training, validation and test sets in the ratio of 60%, 20% and 20%, respectively. This distribution allowed an adequate evaluation of the model's performance, facilitating the optimisation of hyper-parameters and the prevention of overfitting during the validation phase.<br>

The neural network architecture we chose includes: an input layer equipped with 256 neurons and two hidden layers with 128 and 64 neurons, both activated via the ReLU function to introduce non-linearity essential for modelling complex data dynamics. The output layer contains neurons corresponding to the number of unique classes of 'IvaM' and employs a softmax activation function, which facilitates multinomial classification by providing a probability distribution over the possible classes.<br>

The model was fitted with the categorical_crossentropy loss function and optimised using adam, chosen for its efficiency in automatically adjusting the learning rate. Training was performed for 100 epochs with a batch size of 64, using the validation data to monitor the evolution of the model and prevent overfitting.<br>

The final results of the test showed an **accuracy** of 97.40% and a **loss** of 0.1643, indicating a high level of effectiveness of the model in predicting the VAT exemption code from the available information. This high level of accuracy indicates the correct calibration and generalisation of the model to new data. Next, we wanted to check how the model performed as the training epochs progressed, also to check for a possible risk of overfitting. To do this, we printed out two graphs, one inherent to loss and the other for accuracy.<br>

The loss graph shows a rapid drop in loss in both training and validation in the early epochs, indicating that the model has successfully learnt the fundamental relationships in the data. After this initial phase, the loss continues to decrease gradually, stabilising towards the end of the 100 epochs. Interestingly, despite a slight divergence between the training and validation loss after about 20 epochs, the difference remains relatively small, suggesting that the model does not suffer significantly from overfitting.

![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/afdc4999-5807-4b19-8dd3-a440e29d7321) <br>

The accuracy graph reflects a similar trend to that of loss. The accuracy on both sets increases rapidly at the beginning, reaching a plateau. The training accuracy curve shows a slight advantage over the validation accuracy curve, which is common in most machine learning models, but again, the closeness of the two curves suggests that the model generalises well.<br>

![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/dc3898d8-98da-4066-8a64-cfbe1d7f49b8) <br>
The results are as follows:<br>
•**Final Train Loss**: 0.0368<br>
•**Final Train Accuracy**: 98.6798%<br>
•**Final Validation Loss**: 0.1502<br>
•**Final Validation Accuracy**: 97.2761%<br>
The model shows excellent performance, with a balance between learning and generalisation, as demonstrated by the loss and accuracy metrics. These results are very promising and indicate that the model is well trained, with a good ability to generalise to unseen data.<br>

**Why an ANN with 3 layers (256-128-64) and 100 epochs (batch size 64)**
The choice of our model architecture was the result of a rigorous testing process, during which we tested different configurations by varying the number of layers, number of epochs and batch size. This approach allowed us to identify the most effective structure based on key metrics such as accuracy and loss in testing, as well as performance during the training and validation phases. (_If you want to check the results obtained with the different ANN structures, they are present in the file called "ANN_Type_Summary.pdf"_)<br>

To determine the optimal configuration, we ran several models with varying architectures. Each model was evaluated in terms of test accuracy, test loss, and training and validation metrics. This systematic process ensured that our final selection was based on solid data representative of model performance in real-world scenarios.<br>

The model that showed the best performance was the one with three hidden layers of 256, 128, and 64 neurons, respectively, trained for 100 epochs with a batch size of 64. This configuration was shown to offer the best balance between effective learning and generalisation capability, being superior in key metrics. In particular, it achieved high test accuracy while maintaining low loss, indicating good resistance to overfitting as confirmed by validation data.<br>

To select the optimal structure of our model, in addition to considering test accuracy and loss, we paid special attention to the phenomenon of overfitting. We carefully assessed the difference between the training and validation metrics for both loss and accuracy. A small discrepancy between these metrics is indicative of a model that generalises well, while larger differences may signal potential overfitting.<br>

• **Loss discrepancy**: 0.1134 (0.1502 validation - 0.0368 train) <br>
• **Accuracy discrepancy**: 1.4037% (98.6798% train - 97.2761% validation) <br>

Our chosen model, characterised by three hidden levels and trained for 100 epochs with a batch size of 64, stood out not only for its high test accuracy, but also for the smallest discrepancies between the training and validation metrics. This indicates that the model is well balanced, showing good generalisation capability without suffering significantly from overfitting.
The decision to select this model as the 'best' was not only based on accuracy, but also on how effectively the model balances learning with generalisation capability.

<br>

### **3.2 MODEL OPTIMIZATION PROCESS** <br>
Optimising a machine learning model is crucial to improve its performance, ensure good generalisation, reduce the risk of overfitting, and improve computational efficiency. For these reasons, we decided to implement an optimisation of the model, even though we had found it to be very accurate and well-fitted. There are several optimisation methods, we have chosen Early Stopping, but why?<br>

Definition: EarlyStopping throughout training, it keeps an eye on a performance metric, like loss on a validation set, and ends the process if, after a predetermined amount of epochs, performance on this metric no longer improves. EarlyStopping monitors the tracked measure during training, determining whether it gets better with each epoch. Training is terminated if the measurements do not improve after a certain amount of patience epochs. This improves training efficiency and helps avoid overfitting.<br>

The use of early stopping represents an effective strategy for balancing the management of overfitting with the maintenance of high performance. By implementing this technique, the model proves to perform well in terms of both loss and accuracy in training and validation sets, indicating a remarkable ability to generalise without compromising learning. This form of optimisation ensures that the model is robust and generalises effectively to unseen data. It avoids the risk of overfitting the training data, providing robust protection against overfitting while maintaining high performance. Although the use of early stopping may result in a slight decrease in accuracy compared to a non-optimised model, this reduction is minimal and acceptable since the baseline accuracy level is already high.<br>

Early stopping stops training as soon as it is observed that the performance improvement on the validation set ceases, thus ensuring that the model does not just store the training data but actually learns to generalise from them. Given the diversity of VAT codes and the complexity of the task, a model that avoids overfitting to training data is more likely to provide reliable results even on new and varied data.<br>

Early Stopping Model result:<br>
•	**Test Loss**: 0.12497568130493164<br>
•	**Test Accuracy**: 0.9711802005767822<br>
•	**Final Train Loss**: 0.0642<br>
•	**Final Train Accuracy**: 97.8292%<br>
•	**Final Validation Loss**: 0.1278<br>
•	**Final Validation Accuracy**: 97.0344%<br>

Calculating the discrepancies between the training and validation data in our model with early stopping provides important insights into its ability to generalise and resistance to overfitting. The difference between the **final validation and training loss** is 0.0636, while the difference between the **final training and validation accuracy** is 0.79%. These discrepancies indicate that the model shows only a moderate level of overfitting, suggesting a good ability to generalise to new data.<br>
The discrepancy in the loss of 0.0636 reveals that although there is a difference between training and validation performance, it is not excessive, indicating that the model has not overfitted to the training data. Similarly, a discrepancy of about 0.79% in accuracy confirms that the model maintains consistent performance when exposed to new data, a crucial aspect for practical applications. <br>
![image](https://github.com/Pier-Giorgio/BipxTech---ML/assets/151735476/3587447d-2a22-47ba-bbdd-87e11c937625)

In conclusion, the model with early stopping demonstrates a better balance between learning and generalisation than the basic model. Loss and accuracy metrics indicate that the model not only learns effectively from the training data but is also able to transfer this learning effectively to the validation data. This makes it particularly robust and reliable for real-world applications, where generalisation and overfitting reduction are key aspects. <br>

<br>

### **3.3 ANN MODEL WITHOUT VARIABLES CONSISTING OF DESCRIPTIVE TEXT**
During the development of our Artificial Neural Network (ANN) model, we assessed the importance of each of the 26 variables in the dataset, with particular attention to those consisting of descriptive text, such as the ‘DescriptionRow’ column. This column contains phrases referring to names of products, books, activities or projects, which raised doubts as to their actual usefulness for predicting our target.
In order to determine whether these variables were really necessary, we decided to test the model by specifically excluding ‘DescriptionRow’ to observe the impact on performance. The results of this test were significant: the model recorded a Test Loss of 3.3531 and a Test Accuracy of 17.92%. (_We did not include the codes relating to this in the final code as it would have created confusion. If you want to test the veracity of the results, just delete the RowDescrtion column and run the code relating to the model with all the variables present in the .ipynb_)<br>

These results indicate that the removal of the variable ‘DescriptionRow ‘had a significant negative impact on the performance of the model, suggesting that, despite their textual format, the information contained in these descriptions is relevant for the prediction of the exemption VAT code. The high loss and low accuracy compared to previous versions of the model, which included this variable, demonstrate how descriptive phrases may indeed contain key elements necessary for the correct classification and generalisation of the model. This experiment emphasises the importance of carefully considering which variables to exclude in the model optimisation process, especially when dealing with textual data that might seem less directly related to the prediction target.

<br>

### **3.4 RANDOM FOREST MODEL AS A BENCHMAK**
To check the quality of the cleaning and processing of our dataset, we decided to compare the results of our artificial neural network (ANN) model with those of a Random Forest model, used as a benchmark. The choice of the Random Forest model is motivated by its popularity and effectiveness in the field of classification.<br>
The results obtained from the Random Forest model showed an **accuracy** of 98.17%, slightly higher than that of our ANN model. This shows that it was not a fluke that our dataset generated the results shown above. However, we wanted to examine the risk of overfitting in the Random Forest model by comparing the **accuracy on the training set** (99.98%) with that **on the test set** (98.17%).
Despite the high accuracy on both sets, the difference of approximately 1.81% between the training and test accuracy is considered minimal in many contexts and not indicative of **significant overfitting**, especially given the high level of overall accuracy. However, the near perfect training accuracy suggests that the model may have learned specificities of the training data that do not fully translate to the test data.<br>

On the other hand, the closeness between the accuracy on the training, validation and test sets in our ANN model suggests that it is generalising well, without suffering significantly from overfitting. This is indicative of a good model that manages to capture complex relationships between variables, reducing the need for feature engineering. Furthermore, neural networks are known to scale well with large datasets and complex architectures, improving their ability to generalise to new data.

<br>

### **3.5 ANN MODEL WITH 6 VARIABLES**
Once we have decided on the model and verified its performance with our data set, the next step is to implement this model with a user interface that allows users to enter data as input and obtain the corresponding IvaM code as output.<br>
Initially, we planned to use the ANN model with 26 variables as the basis for the user interface. However, this approach requires the user to enter 26 different inputs (corresponding to the 26 variables in our dataset) to derive the IvaM code. This approach would be very inconvenient and inefficient in terms of speed and usability.<br>
To improve efficiency and user experience, we tried to find a model that used fewer variables while maintaining similar accuracy to the 26-variable model. This allows us to significantly reduce the number of inputs required from the user, making the process much smoother and faster, without compromising the accuracy of the IvaM code predictions.
After various analyses, we chose an ANN using the following 6 variables:<br>

1.	**Iva**: we use the Iva variable because it was recommended in the project guide pdf file. The ‘Iva’ column contains the code describing the nature of the transaction or the Iva rate applied.<br>

2.	**Ateco**: this variable is useful for predicting the IvaM code; this usefulness stems mainly from the code's ability to reflect specific tax rules that vary significantly between sectors. It helps to clearly identify which firms are eligible for benefits, such as Iva exemptions or reductions. the ATECO code proves to be a key feature in predictive models such as ANNs, where it helps to significantly improve the accuracy of predictions through the correlation between economic activities and their Iva exemptions<br>

3.	**TIva**: indicating the Iva category or regime applicable to a specific transaction, item or service. classifies transactions according to different Iva regimes or rates, potentially including standard, reduced, zero or exemption rates. This distinction is crucial for correctly calculating the tax due and ensuring tax compliance. This code ensures that invoices are issued with the appropriate Iva rate, influencing the accounting of sales and purchases. It is an important feature in our predictive model, since the type of Iva applied may be related to or influenced by the applicable Iva exemption code.<br>

4.	**Conto**: This variable is crucial for financial reporting and accounting, as it determines the specific account to which each transaction line or invoice is allocated. This helps to categorise and manage financial data accurately. It provides a structured way to understand how transactions are categorised, which can help identify patterns and correlations with Iva exemptions.<br>

5.	**CoDitta**: This code is critical for distinguishing between the various entities involved in transactions, allowing precise attribution of invoices, payments, and other financial transactions to the correct business entity. It acts as a unique key to identify companies within a database or information system. This ensures that all transactions, documents and operations are correctly recorded and attributed to the right entity. This code is then used to link specific invoices, purchase orders, and payments to the correct company.<br>

6.	**Rev**: This variable corresponds to reverse charge is a mechanism that shifts the responsibility for paying Iva from the seller to the buyer in certain transactions. The presence of a specific value in this variable could indicate that for that transaction the Iva should be handled by the buyer rather than the seller. We have chosen this variable because the reverse charge mechanism may influence the likelihood of a transaction being subject to specific Iva exemptions.

This model manages to guarantee an **accuracy** of approximately 95% and a **loss** of 0.2. Regarding performance management to control overfitting risk, we have obtained the following results:<br>
•**Final Train Loss**: 0.1324<br>
•**Final Train Accuracy**: 95.5622%<br>
•**Final Validation Loss**: 0.1716<br>
•**Final Validation Accuracy**: 95.5655%<br>

The 26-variable model outperforms, albeit slightly, the 6-variable model in terms of both accuracy and loss, demonstrating greater effectiveness in generalising to new data. However, the 26-variable model shows a slightly wider margin between training and validation performance, suggesting a potentially greater propensity for overfitting than the 6-variable model. Although the 26-variable model shows superior performance, it needs to be closely monitored for the risk of overfitting, although it currently shows no significant signs of problems.<br>

We also tried to build a model using the 6 variables with the highest correlation rate with IvaM, but this model proved to be inferior in terms of both accuracy and loss at all stages (training, validation and testing). Below are the results of the model with the 6 variables with the highest correlation rate with IvaM ('Art1', 'Iva', 'ContoStd', 'Conto', 'CTra', 'A') (_We did not include the codes relating to this in the final code as it would have created confusion. If you want to test the veracity of the results shown below, just replace the variables in the code present in the .ipynb file_):<br>
•	**Test Loss**: 0.2444<br>
•	**Test Accuracy**: 92.85%<br>
•	**Final Train Loss**: 0.1845<br>
•	**Final Train Accuracy**: 93.65%<br>
•	**Final Validation Loss**: 0.2185<br>
•	**Final Validation Accuracy**: 92.66%<br>

In conclusion, in order to ensure a better user experience in terms of efficiency and convenience, we chose to use the 6-variable model (‘Iva’, ‘Ateco’, ‘TIva’, ‘Conto’, ‘CoDitta’, ‘Rev’) for our purpose. Although it has a lower accuracy than the 26-variable model, the difference is minimal (about 2 percentage points). Furthermore, the 6-variable model performs better in terms of overfitting, ensuring a lower risk of incorrect predictions. The choice of the 6 variables listed above was made through a process of analysis and research into the meaning and usefulness of the 26 variables in our dataset after the cleaning process. This analysis of the variables is documented in the PDF file called ‘BIP_variable_analysis’.

<br>

### **3.6 USER INTERFACE IMPLEMENTATION**
The code inherent to the user interface collects the data entered by the user through interactive widgets, transforms it through encoding and normalisation processes, and uses it to make predictions using a previously trained artificial neural network (ANN) model. The process begins when the user enters values into text fields corresponding to specific attributes such as Iva, Ateco, TIva, Conto, CoDitta and Rev. This data is collected in a DataFrame as soon as the user clicks the ‘Predict’ button.<br>

Subsequently, the categorical attributes between the entered data, specifically ‘Iva’ and ‘Rev’, are converted to numeric formats through a pre-trained encoder. This step is crucial to ensure that the categorical variables are compatible with the model. After encoding, all data are normalised using a pre-trained scaler, ensuring that the values are aligned with the scales used during the model training phase.
Once prepared, the transformed data are provided as input to the ANN model. The model processes these inputs and produces a probability array for each possible class. The class with the highest probability is selected using the function np.argmax(), which identifies the index of the category predicted as most likely. This index is then mapped to a readable label (IvaM code), using a pre-loaded label map, thus transforming the numerical output of the model into a user-understandable form.<br>

Finally, the predicted IvaM code is displayed in an output label, thus showing the user the result of the prediction made by the model. This workflow not only automates the prediction process but also ensures that the user's interactions with the model are intuitive and efficient.

<br>

### **4 CONCLUSIONS**
