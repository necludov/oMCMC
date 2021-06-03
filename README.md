# oMCMC
The repository for the [Orbital MCMC paper](http://arxiv.org/abs/2010.08047).

![visualization][intro_pic]
The illustration of the proposed algorithms. The transparency of a point corresponds to its weight. Red dots correspond to the initial states. From left to right: HMC (stochastically accepts only the last state), Orbital kernel on a periodic orbit (accepts all the states weighted), Orbital kernel on an infinite orbit (accepts the states within a certain region).

## Running scripts

Code is written in __PyTorch and NumPy__. Logger also requires __tabulate and pandas__. Visualizations require __seaborn__.

Scripts should be runned locally from the "scripts" directory simply like (no arguments are specified)
```
python banana.py
```
You should specify the device and all of the settings explicitly in the corresponding scripts. The script prints the results to the terminal through the logger (created by [@senya_ashuha](https://github.com/senya-ashukha)). The logger also automatically saves all of the output into "logger_name.out". The script also saves the estimates of mean and std for different number of gradient evaluations.

__Be careful!__ The output estimates might be quite big: their size is (BATCH SIZE)x(DIMENSION)x(NUMBER OF ITERATIONS). Also, the opt-MC code might be running for a quite long time since it doesn't allows easy batched implementation and runs sequentially.

[intro_pic]: https://github.com/necludov/oMCMC/blob/master/pics/intro.png "Visualizations of algorithms"
