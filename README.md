# About
Single agent auction simulation using deep reinforcement learning algorithm DDPG (Deep Deterministic Policy Gradient).

# Deep RL in Auction Theory 

#### INF - PhD Conclusion Project

Conclusion project of postgraduate program at the Department of Informatics of PUC-Rio.


### Installation and Execution

#### Installing in Anaconda environment

We can use Anaconda to set an environment.

```bash
conda create -n <environment_name> python=3.7.6
conda activate <environment_name>
```


#### Install the dependencies of the project through the command

Then, locate the project's root directory and use pip to install the requirements (`requirements.txt`).

```bash
pip install -r requirements.txt
```

#### To execute the program, just type the following line on the root directory 

```bash
python src/main.py -e <number of episodes> -n <number of players> -b <batch size> -p <ponderated average size> -a <type of auction> -z <number of executions> -s <save test results in a plot> -t <use alert .mp3 file>
```
where the arguments may be passed after the __main.py__ call, as described above, otherwise the default parameters will be selected.
