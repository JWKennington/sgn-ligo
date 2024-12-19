# Contributing Workflow

## Git Branching

The `sgn` team uses the standard git-branch-and-merge workflow, which has brief description
at [GitLab](https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html) and a full description
at [BitBucket](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow). As depicted below,
the workflow involves the creation of new branches for changes, the review of those branches through the Merge Request
process, and then the merging of the new changes into the main branch.

!!! info "Examples Images from GstLAL repo"
    The example images are from the `gstlal` repo, but the process is the same for all repos.

![git-flow](../assets/img/git-flow.png)

### Git Workflow

In general the steps for working with feature branches are:

1. Create a new branch from master: `git checkout -b feature-short-desc`
1. Edit code (and tests)
1. Commit changes: `git commit . -m "comment"`
1. Push branch: `git push origin feature-short-desc`
1. Create merge request on GitLab

## Merge Requests

### Creating a Merge Request

Once you push feature branch, GitLab will prompt on gstlal repo [home page](). Click “Create Merge Request”, or you can
also go to the branches page (Repository > Branches) and select “Merge Request” next to your branch.

![mr-create](../assets/img/mr-create.png)

When creating a merge request:

1. Add short, descriptive title
1. Add description
    - (Uses markdown .md-file style)
    - Summary of additions / changes
    - Describe any tests run (other than CI)
1. Click “Create Merge Request”

![mr-create](../assets/img/mr-create-steps.png)

### Collaborating on merge requests

The Overview page give a general summary of the merge request, including:

1. Link to other page to view changes in detail (read below)
1. Code Review Request
1. Test Suite Status
1. Discussion History
1. Commenting

![mr-overview](../assets/img/mr-overview.png)

#### Leaving a Review

The View Changes page gives a detailed look at the changes made on the feature branch, including:

1. List of files changed
1. Changes
    - Red = removed
    - Green = added
1. Click to leave comment on line
1. Choose “Start a review”

![mr-changes](../assets/img/mr-changes.png)

After review started:

1. comment pending
1. Submit review

![mr-changes](../assets/img/mr-change-submit.png)

#### Responding to Reviews

Reply to code review comments as needed Use “Start a review” to submit all replies at once

![mr-changes](../assets/img/mr-respond.png)

Resolve threads when discussion on a particular piece of code is complete

![mr-changes](../assets/img/mr-resolve.png)

### Merging the Merge Request

Merging:

1. Check all tests passed
1. Check all review comments resolved
1. Check at least one review approval
1. Before clicking “Merge”
    - Check “Delete source branch”
    - Check “Squash commits” if branch history not tidy
1. Click “Merge”
1. Celebrate

![mr-merge](../assets/img/mr-merge.png)
