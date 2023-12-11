#### Discription

Predict flight satisfaction using a decision tree algorithm. This repository includes code for training a decision tree model, evaluating its accuracy, and visualizing key insights such as information gain and correlation matrix.


#### Overview

This repository contains Python code for predicting flight satisfaction using a decision tree algorithm. The decision tree model is trained on a dataset containing information about various flight parameters and passenger satisfaction levels.

#### How to Use

1.  Clone the repository to your local machine.
2.  Install the required dependencies using `pip install -r requirements.txt`.
3.  Run the `main.py` script to execute the flight satisfaction prediction experiment.

    bashCopy code

    `python main.py`

    The experiment is run for two different subsets of data (2000 and 5000 rows) with random state 42.

#### Files and Directories

-   decision_tree.py: Contains the implementation of the DecisionTree class for building and using a decision tree model.
-   flight_satisfaction.csv: Dataset containing information about flight parameters and passenger satisfaction.
-   main.py: The main script to run the flight satisfaction prediction experiment.
-   results/: Directory to store experiment results, including correlation matrix heatmap, distribution plots, and information gain analysis.

#### Results

-   The accuracy of the decision tree model for different subsets of data is printed to the console.
-   Correlation matrix heatmap and distribution plots are saved in the `results/` directory.
-   Information gain analysis results, including a plot, are saved in the `results/rows_{number}/` directory.

### Usage:

1.  Clone the repository:

    bashCopy code

    `git clone https://github.com/your-username/Decision-Tree-Flight-Satisfaction-Predictor.git`

2.  Navigate to the repository directory:

    bashCopy code

    `cd Decision-Tree-Flight-Satisfaction-Predictor`

3.  Install dependencies:

    bashCopy code

    `pip install -r requirements.txt`

4.  Run the experiment:

    bashCopy code

    `python main.py`

### Requirements:

-   Python 3.x
-   pandas
-   numpy
-   scikit-learn
-   matplotlib
-   seaborn
-   tqdm

Feel free to modify the code and experiment with different parameters or datasets. Contributions are welcome!

### License:

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/c/LICENSE) file for details.