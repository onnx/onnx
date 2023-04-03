<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX Community Involvement and Contribution Guidelines

ONNX is a community project and we welcome your contributions! In addition to contributing code, you can also contribute in many other ways:

* Meetings and Discussions
Join SIGS, Working Groups, Community meetings to learn about what is needed and then where there is a good fit to interest and areas of expertise, find ways to actively contribute.  Participate in [ONNX technical discussions](https://github.com/onnx/onnx/discussions) on GitHub.  Join the ONNX Slack channels at LF AI and Data, help answer questions and welcome new members.

* Use Cases and Tools
Develop use cases for ONNX and advocate for ONNX in developer conferences and meetups.  Develop tools that import and export using the ONNX spec, and help grow the community of ONNX users.  Become a champion for ONNX in your company or organization.

* Roadmap and Features
Understand the ONNX roadmap document, feature priorities, and help implement them.  Become an ONNX code and documentation contributor, and work towards committer status on important repos.

* Releases and Model Zoo
Help in achieving a release of ONNX, including increasing the number of models in the ONNX Model Zoo that exercise ONNX features.

* Publications and Blogs
Add to the growing number of arXiv papers that refer to ONNX.  Create blogs, presentations, books, articles and other materials that help increase the adoption of ONNX, and grow the community of users and contributors.

* Steering Committee
Attend ONNX Steering Committee meetings - they are open to all in the community. Help out where needed and appropriate on SC to-do items. Note that SIG and Working Groups leaders as well as others with demonstrated commitment and contributions to ONNX community may want to self-nominate during the annual SC election cycle.

## Contributing code

You can submit a pull request (PR) with your code. The [SIG](community/sigs.md) or [Working Group](community/working-groups.md) that is responsible for the area of the project your PR touches will review it and merge once any comments are addressed.

### DCO
ONNX has adopted the [DCO](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin). All code repositories under ONNX require a DCO. (ONNX previously used a CLA, which is being replaced with the DCO.)

DCO is provided by including a sign-off-by line in commit messages. Using the `-s` flag for `git commit` will automatically append this line. For example, running `git commit -s -m 'commit info.'` it will produce a commit that has the message `commit info. Signed-off-by: My Name <my_email@my_company.com>`. The DCO bot will ensure commits are signed with an email address that matches the commit author before they are eligible to be merged.

If you are using a GUI like the GitHub web site or GitHub Desktop, you'll need to append the `Signed-off-by: My Name <my_email@my_company.com>` manually to each commit message. For the onnx organization [sign-off](https://github.blog/changelog/2022-06-08-admins-can-require-sign-off-on-web-based-commits/) for web based commits is enabled. When this is activated you will see "Sign off and propose changes" instead of "Propose changes" when you are editing files directly at github. It is recommended to set this setting for your own fork as well. Since in the review process commits are made on this fork.

NOTE: the sign-off is needed for each commit in the PR, not at the PR level.

If you have old commits that are not signed, use the following commands to squash the old PR (original branch) into a single commit. This is an easier way to signoff old commits in old PR.

```bash
git checkout main
git checkout -b temporary_patch              # create a new branch as temporary
git merge --squash original_patch            # copy from old branch
git branch -d original_patch                 # remove old branch
git checkout -b original_patch               # create a new branch with the same name (override)
git commit -m 'type your own commit msg' -s  # signoff that single commit
git push origin original_patch -f            # forcibly override the old branch`
```
