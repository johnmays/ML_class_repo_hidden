## How to run John Mays' code:

### First, you need data to run it on.
- Start a command prompt
- Navigate to the root directory in our repository
- create a directory titled `data`
- on our repository's readme, there is a link to download the data that we used (Musk1.csv, Elephant.csv, etc.)
- simply place any CSVs you would like to test on in the aforementioned `data` folder. (the only acceptable data format for our code is .csv, and it must be a multiple-instance problem dataset)

### Next, you must run it
- Make sure you are still in our root directory
- While there, activate poetry with the command `poetry shell` (you may have to run `poetry install` the first time around)
    - This poetry assembly is _very_ similar to the one we were given for the semester.
- Navigate to the `root/code` directory in our repository.  
- run one of my algorithms:
    - start your command with exactly this: `python3 jkm100/main.py`
    - follow it with the required arguments:
        - `folder`, which should be `../data` if you followed my instructions.
        - `datasets`, which should either be a comma-separated list of datasets you would like to run on, or just one with no commas
    - then, any additional argumens
        - `-a [algorithm]` is where you specify if you'd like to run a specific algorithm.  If left blank, all three will run on your dataset(s).
        - acceptable arguments are `apr`, `apr_extension`, and `sbMIL`
    - then, any flags
        - give the `--allpos` flag only if you specified `-a` as `apr`.  Specifying runs APR with all-positive hypothesis.
        - give the `--no-cv` flag if you'd like to fit and predict on the same set of data with no cv split or anything.
- __once you run, there should be command line output to let you know the algorithm is running__
    
### Some Example Calls:

- Run all positive version of APR algo on Elephants dataset: `python3 jkm100/main.py ../data Elephant.csv -a apr --allpos`
- Run all three algorithms on Musk1: `python3 jkm100/main.py ../data Musk1.csv`
- Run sbMIL on Musk2.csv with no cross validation: `python3 jkm100/main.py ../data Musk2.csv -a sbMIL --no-cv`

## A note on runtimes:
Some of my algorithms run faster than others.  Musk1 is a small dataset that typically takes very little time to run.  sbMIL is typically faster than APR.  None of my algorithms have ever taken greater than twenty minutes to run on one of my datasets, although three in conjunction could easily go over twenty minutes on my GPU-less laptop.


