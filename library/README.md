# Course Library
This directory has some useful materials for your own learning. Feel free to contribute to this place with Pull Request!

## Git 101
You can find some useful commands [here](https://git-scm.com/cheat-sheet). Git Bash is your good friend!
Here are some commands you will use for this class a lot:
- **clone** a repositry: when you want to clone a repositry from GitHub, like this class with URL: *https://github.com/yunmeow5566/RSD-MML-Pilot*, you can clone it to your local folder by typing (or copying from the website)
	`git clone https://github.com/yunmeow5566/RSD-MML-Pilot.git`

- **pull** latest changes: if you have cloned a repository on your local, and you want to get the latest version. You will need:
`git fetch origin` to connect to the repository on GitHub to get all changes.
`git pull` to apply changes to your local repository.


- **create** a new branch: remember to `cd <repository name>` under your Git Bash to enter the repository. Then you can type
`git checkout -b <a new branch name>` to create a new branch.

- **switch** between branches: you will need to commit your changes to the current branch before you switch. Type
`git checkout <an existing branch name>` to jump to your target branch

- **Stage** changes: you can add all changes or part of changes into each commit.
This command will add all changed files: `git add -A`
Alternatively, you can add changed files by their names: `git add <path to files>`

- **Status** check: this command helps us to know where we are:
`git status`

- **commit** staged changes: you can wrap multiple changes in one commit with commit message as:
`git commit -m "your message"`

- **push** commits: after your have commits in your local repository, you can type
`git push` to move them to GitHub. Notice that for the first time, you will need to type
`git push origin <branch name>` to specify the branch name. 

**Note** you can run `git config --global push.autoSetupRemote true` once, then `git push` will automatically create a new origin branch based on your branch name.

## Git Collaboration
When multiple developers work on the same repository, they will need to deal with merge conflicts.
Why does merge conflict happen? Usuaully there is a *main* branch where all team members need to submit Pull Requests to make changes.
Consider this is a commit trace:
	> --A--B--C

Developer A branched out from commit A, and made changes on a few files and submitted a pull request.
Meanwhile, another Developer B merged his pull request to the main branch, which touched files affected by A's pull request.
Then Developer A needs to take B's changes (commit B) to his local, resolve conflicts and update his branch/pull request.
### Situation 1: All of your commits are on another (feature) branch.
You will hit this situation if you always branch out from *main* before making commits.
For this case, you will firstly go back to the *main* branch (type `git checkout main`) and then pull changes (type `git pull`).
Then you can switch to your feature branch again (type `git checkout <your branch>`).

Now you could do either `git rebase main` or `git merge main`. I personally prefer **rebase** as it keeps commit history clean.
Once you type `git rebase main` in your feature branch, you will see git is trying to resolve conflicts for you, but some files will be marked as "CONFLICT" in your git bash. 

For most of the situations, you will need to visit each CONFLICT file, review differences between the two versions and edit.
Once you are done with one file, type `git add <file path>`, and keep working on other files.

After you are done, type `git rebase --continue`, you might be prompted to edit a commit message, just exit the editing mode.
If you have too many commits on your feature branch, you might want to do `git rebase -i HEAD~<number of commits you want to work on>`, and squash commits before rebase to *main*.
Alternatively, you can use merge.

### Situation 2: You accidentally put your feature commits on main
First all, you should not, and cannot push these changes to *main* directly.
You will need to create a branch from your local repository first by typing `git checkout -b <a new branch name>`.

Now you have these feature commits in your friendly local feature branch. Then checkout *main*. If you try to pull directly, you might see complaints.
What you need to do is to clean up your local *main*.
You can type `git fetch origin` and `git reset --hard origin/main` to match your local *main* to the upstream/shared repository.

Once your local main is the same as the upstream version, you can follow steps in **Situation 1** to resolve conflicts.
