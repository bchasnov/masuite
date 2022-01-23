# Analysis
The analysis folder contains the files used to alayze the data sets and generate graphs. Currently, each file is based on Gridhunter data.

* `dataplotting.ipnyb` - Creates a plot based from a trial of data. Also can make multiple 2x2 graphs of the returns and losses iterating over all of the data, or all of one type of data (returns or losses) from every trial into one graph.  

* `raw_df_extractor` - Creates a full pandas dataframe from all of the data. raw_df contains the column title information from the .csv files. 

* `score.ipynb` - This file uses a scoring function that is based on "Agent Wins / Total Time" Creates 4 pandas dataframes of the scores by looping through all of the .csv files obtained from running an experiment. big_score_df contains all the data with the difference in rewards. score_df contains the scoring percentage after each trial. batch_df contains the scores for each batch. epoch_df contains the score perfcentage after each epoch. 

* `eloscore.ipynb` - This file creates a dataframe of ELO scores based on tournament data, and ranks each player with their corresponding ELO.
* ELO score based on
* https://reader.elsevier.com/reader/sd/pii/S0169204616301165?token=00473F442A78E7A4B15546C8E4C0AD6DD5BAC64B2EBDBDA4E44B3BE0F82A9D7DF7E3887DC0576E157B7F5BC5E68B4EC3&originRegion=us-east-1&originCreation=20211120194522![image](https://user-images.githubusercontent.com/50932746/150702320-3c13ef80-f727-411f-b88a-3257e9593c20.png)

Formula used to determine ELO Score is as follows:
![equation]<img src="https://latex.codecogs.com/svg.image?{R_{i}}^{'}&space;=&space;R_{i}&space;&plus;&space;K(S_{i}-E_{i})" title="{R_{i}}^{'} = R_{i} + K(S_{i}-E_{i})" />
Ri' = New Score
Ri = Original Score
K = Constant that controls magntiude of score changes
Si = Outcome (Lose = 0, Draw = 0.5, Win = 1)
Ei = Expected Outcome (based on https://arxiv.org/pdf/1908.09213.pdf)
![equation]<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;E_{i}&space;=&space;\frac{1}{1&plus;10^{\frac{(S_{1}-S_{2})}{2C}}" title="E_{i} = \frac{1}{1+10^{\frac{(S_{1}-S_{2})}{2C}}" />
S1 = Agent 1
S2 = Agent 2
C = Assumed class interval which determines distribution of scores,

# Improvements to make

* `dataplotting.ipnyb` - defines the seed for any randomness in the enviornment.

* `raw_df_extractor` - Allow the raw_df to get the column title information automatically than manually writing it in. Remove the grad_norms_to_float part of the data frame.

* `score.ipynb` - Use the raw_df_extractor function to get the total data frame, then make the data frames.

# Current Data Analysis Files

* `Gridhunter_data.ipnyb` - Using these functions, analyzes the data for a Gridhunter game.

* `Soccergrid_data.ipnyb` - Using these functions, analyzes the data for a soccer game.

* `Prisoners_data.ipynb` - Using these functions, analyzes the data for an iterated prisoner's dilemma game. *INCOMPLETE*
