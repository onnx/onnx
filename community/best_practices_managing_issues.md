# Best practices for managing onnx issues

## Enforce templates
Currently there are no templates for creating issues in onnx. This makes it very difficult for community members to triage issues. Going forward templates shall be added and maintained for ONNX issues.

Following templates shall be added
* ONNX Bug
* ONNX Question
* Propose New Feature, Operator or Enhancement


## Label issues

* Going forward all issues must be labeled. 
* Multiple labels should be used to properly capture the issue. For example when a user logs a shape inference bug for a particular operator.  [bug][shape-inference] labels should be added.
* Labels should use a consistent naming scheme
* New label must be created for each release and it must be applied to all PRs and issues which are targeted for the release.
* "backlog" label should be applied to all the issues which are cut from current release but are still on the radar to be addressed. This will prevent them from being closed as stale.

### Rules for label naming scheme

* Only lower case alphabets must be used
* Whenever there are multiple words they must be separated by a hyphen '-'.
* Names must be descriptive
* Label for release must begin with "rel-"

### Label tags

Following labels will be added. [Some of these labels already exists]
* bug
* question
* enhancement
* tech-debt
* needs-triage
* good-first-issue
* backlog
* operators
* build
* documentation
* test
* spec
* converters
* version_converter
* training


## Close abandoned issues

Onnx shall use Probot/stale to close any abandoned issues.

### Rules for closing abandoned issues

* After a 90 days of inactivity, a label will be applied to mark an issue as stale, and a comment will be posted to notify contributors that the Issue will be closed.
* If the Issue is updated, or anyone comments, then the stale label is removed and nothing further is done until it becomes stale again.
* If no more activity occurs, the Issue will be automatically closed with a comment.
* Following issues shall be exempted from auto-closing
    * Release labels. These labels start with "rel-"
    * backlog label
