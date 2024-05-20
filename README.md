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

<br>
<br>
<br>











<br>

### **ENCODING AND NORMALIZATION**
In this section we will describe the process of encoding and normalizing the data to prepare it for training the predictive model. These steps are critical to ensuring that the model can effectively learn from available features.<br>

We have chosen to use Ordinal Encoding to convert the categorical variables to numeric in our dataset. This technique assigns a unique integer value to each category, preserving the natural order of the categories (when it exists). The choice of Ordinal Encoding is primarily motivated by its simplicity and efficiency, as it is easier to implement and occupies less memory than one-hot encoding, which is ideal for working with large datasets. In addition, Ordinal Encoding is particularly compatible with certain machine learning algorithms that can benefit from the use of ordinal variables where the order of the categories could have an important meaning.<br>

To do this, we selected all non-numeric columns in the dataset, then to ensure compatibility with the encoder, all non-numeric columns were converted into strings and finally the categorical columns were transformed into ordinal values using the ordinal encoder.<br>

Concerning the normalisation process (normaliser Min-Max), we chose to scale the numerical variables in a range between 0 and 1. This process was crucial to ensure that all features have equal importance during the training of the model, preventing certain features from dominating due to their different scales. We chose to normalise rather than standardise because the distributions of our data did not follow a normal distribution. This step is particularly useful for models that are sensitive to feature scales, such as models based on neural networks, and can improve the speed and stability of convergence.


<br>

### **MODEL IMPLEMENTATION - ANN with 3 layers (256-128-64), 100 epochs, batch size 64**
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

### **MODEL OPTIMIZATION PROCESS** <br>
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

### **ANN MODEL WITHOUT VARIABLES CONSISTING OF DESCRIPTIVE TEXT**
During the development of our Artificial Neural Network (ANN) model, we assessed the importance of each of the 26 variables in the dataset, with particular attention to those consisting of descriptive text, such as the ‘DescriptionRow’ column. This column contains phrases referring to names of products, books, activities or projects, which raised doubts as to their actual usefulness for predicting our target.
In order to determine whether these variables were really necessary, we decided to test the model by specifically excluding ‘DescriptionRow’ to observe the impact on performance. The results of this test were significant: the model recorded a Test Loss of 3.3531 and a Test Accuracy of 17.92%. (_We did not include the codes relating to this in the final code as it would have created confusion. If you want to test the veracity of the results, just delete the RowDescrtion column and run the code relating to the model with all the variables present in the .ipynb_)<br>

These results indicate that the removal of the variable ‘DescriptionRow ‘had a significant negative impact on the performance of the model, suggesting that, despite their textual format, the information contained in these descriptions is relevant for the prediction of the exemption VAT code. The high loss and low accuracy compared to previous versions of the model, which included this variable, demonstrate how descriptive phrases may indeed contain key elements necessary for the correct classification and generalisation of the model. This experiment emphasises the importance of carefully considering which variables to exclude in the model optimisation process, especially when dealing with textual data that might seem less directly related to the prediction target.

<br>

### **RANDOM FOREST MODEL AS A BENCHMAK**
To check the quality of the cleaning and processing of our dataset, we decided to compare the results of our artificial neural network (ANN) model with those of a Random Forest model, used as a benchmark. The choice of the Random Forest model is motivated by its popularity and effectiveness in the field of classification.<br>
The results obtained from the Random Forest model showed an **accuracy** of 98.17%, slightly higher than that of our ANN model. This shows that it was not a fluke that our dataset generated the results shown above. However, we wanted to examine the risk of overfitting in the Random Forest model by comparing the **accuracy on the training set** (99.98%) with that **on the test set** (98.17%).
Despite the high accuracy on both sets, the difference of approximately 1.81% between the training and test accuracy is considered minimal in many contexts and not indicative of **significant overfitting**, especially given the high level of overall accuracy. However, the near perfect training accuracy suggests that the model may have learned specificities of the training data that do not fully translate to the test data.<br>

On the other hand, the closeness between the accuracy on the training, validation and test sets in our ANN model suggests that it is generalising well, without suffering significantly from overfitting. This is indicative of a good model that manages to capture complex relationships between variables, reducing the need for feature engineering. Furthermore, neural networks are known to scale well with large datasets and complex architectures, improving their ability to generalise to new data.

<br>

### **ANN MODEL WITH 6 VARIABLES**
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

### **USER INTERFACE IMPLEMENTATION**
The code inherent to the user interface collects the data entered by the user through interactive widgets, transforms it through encoding and normalisation processes, and uses it to make predictions using a previously trained artificial neural network (ANN) model. The process begins when the user enters values into text fields corresponding to specific attributes such as Iva, Ateco, TIva, Conto, CoDitta and Rev. This data is collected in a DataFrame as soon as the user clicks the ‘Predict’ button.<br>

Subsequently, the categorical attributes between the entered data, specifically ‘Iva’ and ‘Rev’, are converted to numeric formats through a pre-trained encoder. This step is crucial to ensure that the categorical variables are compatible with the model. After encoding, all data are normalised using a pre-trained scaler, ensuring that the values are aligned with the scales used during the model training phase.
Once prepared, the transformed data are provided as input to the ANN model. The model processes these inputs and produces a probability array for each possible class. The class with the highest probability is selected using the function np.argmax(), which identifies the index of the category predicted as most likely. This index is then mapped to a readable label (IvaM code), using a pre-loaded label map, thus transforming the numerical output of the model into a user-understandable form.<br>

Finally, the predicted IvaM code is displayed in an output label, thus showing the user the result of the prediction made by the model. This workflow not only automates the prediction process but also ensures that the user's interactions with the model are intuitive and efficient.

<br>

### **CONCLUSIONS**
