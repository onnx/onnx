# Best practices for managing onnx issues
These best practices apply to all onnx ecosystem repos under SIG governance.

## Enforce templates
Templates shall be added and maintained for creating Issues.

Following templates shall be added
* Report a bug
* Ask a question
* Propose a new feature or other enhancement


## Label issues

* All issues must be labeled.
* Multiple labels should be used to properly capture the issue. For example, when a user logs a shape inference bug for a particular operator.  [bug][shape-inference] labels should be added.
* Labels should use a consistent naming scheme.

### Rules for label naming scheme

* Only lower case alphabets must be used.
* Whenever there are multiple words they must be separated by a space ' '.
* Names must be descriptive.

### Label tags

Along with other repo specific labels following labels shall be added.
* bug
* question
* enhancement
* good first issue
* documentation


## Close abandoned issues

* All abandoned Issues shall be closed.

### Rules for closing abandoned issues

* After 90 days of inactivity, a label will be applied to mark an issue as stale, and a comment will be posted to notify contributors that the Issue will be closed.
* If the Issue is updated, or anyone comments, then the stale label is removed and nothing further is done until it becomes stale again.
* If no more activity occurs, the Issue will be automatically closed with a comment.
* Issues in a milestone will be excluded from auto-closing.
