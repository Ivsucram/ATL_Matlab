# Reference

[ArXiv -> ATL: Autonomous Knowledge Transfer from Many Streaming Processes](https://arxiv.org/abs/1910.03434)

[ResearchGate -> ATL: Autonomous Knowledge Transfer from Many Streaming Processes](https://www.researchgate.net/publication/336361712_ATL_Autonomous_Knowledge_Transfer_from_Many_Streaming_Processes)

# Notes

`ATL_Matlab` was the original code used for this paper. In the following weeks, we will upload a version of this code on Python (Pytorch, with autograd enabled and GPU support). This will make it easier for researchers to use ATL in their benchmarks.

Update: You can find [ATL_Python](https://github.com/Ivsucram/ATL_Python) already by following [this link](https://github.com/Ivsucram/ATL_Python).

# ATL_Matlab

ATL: Autonomous Knowledge Transfer From Many Streaming Processes
ACM CIKM 2019

1. Clone `ATL_Matlab` git to your computer, or just download the files.

2. Provide a dataset by replacing the file `data.csv`
The current `data.csv` holds [https://www.researchgate.net/publication/221653408_A_Streaming_Ensemble_Algorithm_SEA_for_Large-Scale_Classification](SEA) dataset.
`data.csv` must be prepared as following:

```
- Each row presents a new data sample
- Each column presents a data feature
- The last column presents the label for that sample. Don't use one-hot encoding. Use a format from 1 onwards
```

3. Open Matlab. The code was developed using Matlab 2018b, so if you use an older version, you might get some incompability errors.

You can use Matlab 2018b or newer.
Matlab may prompt you to install some official add-ons, as:

```
- Deep Learning Toolbox
- Fuzzy Logic Toolbox
- Digital Processing Signal Toolbox
```

4. Inside Matlab, travel until the folder where you downloaded `ATL_Matlab`.

5. On the Matlab terminal, just type `ATL`. This will execute ATL, which will read your data.csv and process it.

ATL will automatically normalize your data and split your data into 2 streams (Source and Target data streams) with a bias between them, as described in the paper.

Matlab will print ATL status at the end of every minibatch, where you will be able to follow useful information as:

```
- Training time (maximum, mean, minimum, current and accumulated)
- Testing time (maximum, mean, minimum, current and accumulated)
- The number of GMM clusters (maximum, mean, minimum and current)
- The target classification rate
- And a quick review of ATL structure (both discriminative and generative phases), where you can see how many automatically generated nodes were created.
```

At the end of the process, Matlab will plot 6 graphs:

```
- Network bias and Network variance w.r.t. the generative phase
- Network bias and Network variance w.r.t. the discriminative phase
- The target and source classification rate evolution, as well as the final mean accuracy of the network
- All losses over time, and how they influence the network learning
- The evolution of GMMs on Source and Taret AGMMs over time
- The processing time per mini-batch and the total processing time as well, both for training and testing
```

Thank you.

# Download all datasets used on the paper

As some datasets are too big, we can't upload them to GitHub. GitHub has a size limite of 35MB per file. Because of that, you can find all the datasets in a csv format on the anonymous link below. To test it, copy the desired dataset to the same foler as ATL and rename it to `data.csv`.

- [https://drive.google.com/open?id=1emgVw6muSodzozQcuz7ks8XeZYPxGEZ7](https://drive.google.com/open?id=1emgVw6muSodzozQcuz7ks8XeZYPxGEZ7)



