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

We will add more tips when we need to deal with merge conflicts.