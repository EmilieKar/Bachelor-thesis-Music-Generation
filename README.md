# Bachelor thesis: Music generation
This repository consists of the documents and code belonging to bachelor thesis on music generation with genetic algorithms done VT2021. I've reuploaded the repository since the original one contains too much private information for the project members. 

Group members: 
- [Zonotara](https://github.com/Zonotora)
- [viklofg](https://github.com/viklofg)
- [pudkipz](https://github.com/pudkipz)
- [streckgumman](https://github.com/streckgumman)
- [mabergst](https://github.com/mabergst)
- [EmilieKar](https://github.com/EmilieKar)

This project was a continuation on a course project on [generating fugues](https://github.com/EmilieKar/Fugue_Generator_v2). A big focus of this project was to generate new and original music NOT generate music trained on large corpuses. 


We used genetic algorithms with several different methods of evaluating the music pieces. 
- Automatic evaluation: acoording to predetermined desierable criteria. 
- Automatic evaluation: by a neural network trained on different genres of classical music. 
- Interactive evaluation: at certain time steps humans listened to a set number of pieces and graded them with a numerical score. 
- Semi-interactive: using human scores as trainng data for a Neural Network model to customize the sutomatic training to individual preferences. 

In addition to this we developed a number of proposed datastructures to represent the music that were designed for the genetic algorithms specifically. 

For more information please look at the final thesis paper (Thesis_report.pdf) included in the repository. 
- - -

## Setup

Setup the project with the following
```
pip install -e .
```
Make sure you run this command from the root directory, that is, the folder with `setup.py`. Be aware of the dot after `-e`.

This will install EvolutionaryMusic in editable mode. Any changes made to the code will be immediately applied across the system.

## Testing

We used [Pytest](https://docs.pytest.org/en/stable/) for testing.
```
pip install pytest
```
To run all tests manually
```
pytest -v tests
```
Make sure you run this command from the root directory, `pytest -v <path>`. Alternatively, you could just run `pytest -v` because pytest will recursively find all files with `test_*`.


## Codestyle

[Flake8](https://flake8.pycqa.org/en/latest/) will be used as linter for the project. See [rules](https://www.flake8rules.com/).
```
pip install flake8
```

[Black](https://github.com/psf/black) is used as formatter
```
pip install black
```

We use the built in tools for both in vscode

## Github worklflow
( The policies we had working together long distance during the pandemic. Good guides for people in the group who hadn't previously worked a lot with Github. Left them here for future reference or for anyone who might find it helpful for their project. )

- the owner of the commit is responsible for merging the pull request after approval and for solving any arising conflicts
- commits have to be checked by another team member before being commited to the main branch
- aim for 4-6 commits per pull request
- make clear commits and try to implement smaller parts and do pull requests often
- When reviewing other pull requests add comments but not request changes so other people may approve the request
- when approving make sure to check all previous comments and that all files (and tests) are runnable

### Committing


Make sure you only commit relevant files to avoid uneccessary additional commits. Write descriptive commits in english and avoid commits like: `Fix stuff`, `Small changes` etc.
```
git add <file>
```
or
```
git add <file> <file> <file>
```

### Continue working on code related to existing pull request


Begin to checkout the branch which is awaiting pull request review
```
git checkout <branch-name>
```
then create a new branch
```
git checkout -b <new-branch-name>
```
This will create a new branch from the dependent branch. If the dependent branch changes while waiting to get merged, rebase the newly created branch to keep it up-to-date
```
git checkout <new-branch-name>
git rebase origin <branch-name>
```
When the dependent branch is merged, do the following to get a "usual" branch
```
git checkout master
git pull origin master
git checkout <new-branch-name>
git rebase --onto master <branch-name> <new-branch-name>
```

See [this](https://softwareengineering.stackexchange.com/questions/351727/working-on-a-branch-with-a-dependence-on-another-branch-that-is-being-reviewed) for more information.


### Keeping current branch up-to-date

If you want to get all the recent changes from master, do the following

```
git fetch origin
```
Make sure you are on the right branch
```
git checkout <branch-name>
```
Then
```
git rebase origin/master
```

If you get merge conflicts see [Solving merge conflicts](#solving-merge-conflicts).


### Solving merge conflicts

If you have merge conflicts using `git rebase` then you will go through every commit until you get to `HEAD`. You will stop at each conflict and need to correct them manually
```
<<<<< (Current change)

# some changes ...

=====

# some other changes ...

>>>>> (Incoming change)
```
You have to decide which one is relevant. After you have resolved all conflicts
```
git add/rm <conflicted-files>
git rebase --continue
```
You then have to correct conflicted files once again if `git` stops at some other commits, else you are good to go.


### Fixing conflicts after approved pull request

If there are conflicts after approval you have resolve these. Begin to
```
git fetch origin
git checkout <branch-name>
git rebase master
```
This will rebase (replay every commit on top of master) the approved branch onto master. You will have to solve any arising conflicts manually (See [Solving merge conflicts](#solving-merge-conflicts)). Once you have resolved every conflict, do

```
git push --force-with-lease origin <branch-name>
```
Be careful, because this will overwrite changes on `<branch-name>` so make sure you only resolve conflicts. Once you have pushed you should be able to rebase and merge with master.



