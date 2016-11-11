import os
import pandas as pd

os.chdir("F:\Data Scientist\Python\Project\all csv")
#reads all the csv file from a directory
for filename in os.listdir("."):
    #read each csv file as pandas dataframe
    os.chdir("\F:\Data Scientist\Python\Project\all csv")
    df = pd.read_csv(filename)
    #threshold number of colomns with null values set as 80%
    frac = len(df) * 0.8
    colomns with number of null values more than 80% removed
    #variables with more than 20% na's removed
    df = df.dropna(thresh=frac, axis=1)
    #list all the colomns
    stri=str(df.isnull().sum())
    #write in txt file
    os.chdir("F:\Data Scientist\Python\Project\new")
    with open("file.txt","a") as f:
        f.write(filename+"\n")
        f.write(stri+"\n")
        f.write("\n")
