<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Open Governance

## TL;DR

ONNX is rolling out open governance to encourage broader participation beyond the founding companies. We hope this will make the decision making process more transparent, enable better technical decisions with consideration of more viewpoints, and share the work of maintenance. We want ONNX to be the standard the whole community rallies to without reservations.

ONNX open governance creates 3 roles: Member, Contributor and Approver. 3 structures are also created: Steering Committee, Special Interest Groups (SIGs), Working Groups. Contributors and Approvers can vote for the Steering Committee members. The Steering Committee charters SIGs and appoints SIG chairs. Every piece of ONNX belongs to some SIG. Contributors and Approvers participate in one or more SIGs. Our governance structure is based on the successful model of Kubernetes.

The effort is bootstrapped with an initial Steering Committee and set of SIGs with the first elections to occur after 1 year.

## Principles

The ONNX community adheres to the following principles:

* __Open__: ONNX is open source. See repository guidelines and DCO, below.
* __Welcoming and respectful__: See Code of Conduct, below.
* __Transparent and accessible__: Work and collaboration should be done in public. See SIG governance, below.
* __Merit__: Ideas and contributions are accepted according to their technical merit and alignment with project objectives, scope and design principles. Engineering investment >> corporate sponsorship
* __Speed__: Contributing the time and effort to ensure fast decision-making is key to ensuring that the specifications produced is aligned to the fast iteration of machine learning technologies.

## Community Roles

### Members
Members are individuals who are interested in or participate in the ONNX community. Members are able to follow and participate in all public modes of communication used by the ONNX community including but not limited to GitHub, Slack, Stack Overflow, email announcements and discussion aliases. Members are expected to adhere to the Code of Conduct but do not have any specific responsibilities.

### Contributors
Contributors are Members who are active contributors to the community. They can have issues and PRs assigned to them. They also have voting privileges. Contributors can be active in many ways including but not limited to:

* Authoring or reviewing PRs on GitHub
* Filing or commenting on issues on GitHub
* Contributing to SIG, subproject, or community discussions (e.g. Slack, meetings, email discussion forums, Stack Overflow, etc)
* Creator of content, promoting and advocating the ONNX specification

A Member can become a Contributor by being sponsored by 2 existing Approvers from different companies. Contributors who are not active in the last 12 months will be removed.

### Approvers
Approvers are Contributors who are experienced with some aspect of the project and with general software engineering principles. Approvers are responsible for reviewing contributions for acceptance by considering not just code quality but also holistic impact of the contribution including compatibility, performance, and interactions with other areas.

Approvers need to be active Contributors for at least 3 months and be sponsored by a SIG chair with no objections from other SIG chairs.

### Member Companies
Member Companies are organizations that support ONNX in one or more of the following ways:
* Having employees participate in SIGs, Working Groups, or the Steering Committee
* Hosting a workshop or meetup for ONNX
* Providing resources for building or hosting ONNX assets
* Doing media or PR activities to promote ONNX
* Shipping a product that supports ONNX

Member Companies do not have any voting rights, except via their employees who are Contributors. Affiliates and subsidiaries are considered part of the Member Company and not as separate organizations. Being a Member Company does not by itself confer any compliance or certification to the Member Company's products.

Member Companies can request their logo be displayed on the website and other materials by following these [instructions](logo_request.md).

## Organizational Structure

The ONNX community is organized in the following manner, with all governance and execution being planned and coordinated as follows:

* **Steering Committee** is made up of a set number of people whose charter it is to define and iterate on the vision, goals, and governance process of the ONNX community.
* **Special Interest Groups (SIGs)** are persistent groups that are responsible for specific parts of the project. SIGs must have open and transparent proceedings. Anyone is welcome to participate and contribute provided they follow the Code of Conduct. The purpose of a SIG is to develop a set of goals to be achieved over a set period of time, and then to gather input, drive consensus and closure, implement code contributions, and other related activities to achieve the goal. SIGs are also responsible for ongoing maintenance of the code in their areas.
* **Working Groups** are temporary groups that are formed to address issues that cross SIG boundaries. Working groups do not own any code ownership or other long term artifacts. Working groups can report back and act through involved SIGs.

### Steering Committee

#### Role

The Steering Committee has a set of rights and responsibilities including the following:

* Define, evolve, and defend the vision, values, mission, and scope of the project.
* Define, evolve, and defend a Code of Conduct,  which must include a neutral, unbiased process for resolving conflicts.
* Define and evolve project governance structures and policies, including how members become contributors, approvers, SIG chairs, etc.
* Charter and refine policy for defining new community groups (Special Interest Groups, Working Groups, and any future possible defined structure), and establish transparency and accountability policies for such groups.
* Decide, for the purpose of elections, who is a  member of standing of the ONNX project, and what privileges that entails.
* Decide which functional areas and scope are part of the ONNX project, including accepting new or pruning old SIGs and Working Groups.
* Decide how and when official releases of ONNX artifacts are made and what they include.
* Declare releases when quality/feature/other requirements are met.
* Control access to, establish processes regarding, and provide a final escalation path for any ONNX repository, which currently includes all repositories under the ONNX GitHub organizations
* Control and delegate access to and establish processes regarding other project resources/assets, including artifact repositories, build and test infrastructure, web sites and their domains, blogs, social-media accounts, etc.
* Define any certification process.
* Manage the ONNX brand and any outbound marketing.
* Make decisions by majority vote if consensus cannot be reached.

#### Structure

The Steering Committee consists of 5 individuals. No single Member Company may have more than 1 representative. Members serve 1 year terms.

The starting composition will be individuals from Microsoft, Facebook, Amazon, and 2 other Member Companies, who have been picked by the three founding members based on contributions and experience.

After the initial term of each Steering Committee representative is completed, their seat will be open for any contributor in the community to be elected into the seat via a community vote. Only contributors may vote, but would be restricted to one vote per Member Company. Therefore, if a Member Company had three contributors in good standing, the three contributors would have to select who would vote on their behalf.

If a member of the Steering Committee changes companies, by default they retain and may continue on with the role. If the employment change results in a single Member Company having more than one representative, then one of them must resign. When there is a vacancy on the Steering Committee, the remaining members can appoint a new representative for the remainder of the term until the next election.

The Steering Committee will decide on and publish an election process within 3 months of formalizing this organizational structure. This will cover voting eligibility, eligibility for candidacy, election process and schedule. During this time period, the Steering Committee will also establish SIGs and Working Groups.

A Steering Committee member can be removed due to Code of Conduct violations.

### SIG - Special Interest Groups

#### Role

The ONNX project is organized primarily into Special Interest Groups, or SIGs. Each SIG is comprised of individuals from multiple companies and organizations, with a common purpose of advancing the project with respect to a specific topic.

Our goal is to enable a distributed decision structure and code ownership, as well as providing focused forums for getting work done, making decisions, and on-boarding new contributors. Every identifiable part of the project (e.g., repository, subdirectory, API, test, issue, PR, Slack channel) is intended to be owned by some SIG. At the time of inception of this organizational structure, the following SIGs will be present:

* Architecture & Infra
    * This SIG is responsible for defining and maintaining the core ONNX format, the build and CI/CD systems for ONNX repositories, publishing release packages for ONNX, and creating tools to help integrate with and test against the ONNX standard. This SIG is also the defacto owner of files in the main ONNX repository unless explicitly owned by another SIG.
* Operator Standardization
    * This SIG is responsible for determining the operators that are part of the ONNX spec (ONNX and ONNX-ML domains), ensuring high quality operator definitions and documentation, establishing criteria for adding new operators, managing ops domains and compliance tiers, and enforcing versioning mechanisms.
* Converters
    * This SIG is responsible for developing and maintaining the various converter repositories under ONNX.
* Model zoo and tutorials
    * This SIG is responsible for the respective repositories with the charter of providing a comprehensive collection of state of the art ONNX models from a variety of sources and making it easy for users to get started with ONNX and the ecosystem around it.

#### Structure

SIGs must have at least one, and may have up to two SIG chairs at any given time. SIG chairs are intended to be organizers and facilitators, responsible for the operation of the SIG and for communication and coordination with the other SIGs, the Steering Committee, and the broader community. All SIG chairs are appointed by the Steering Committee. If there are more than two contributors being considered for a particular SIG, the Steering Committee will vote on and resolve who the chairs would be. Candidates need to be Approvers.

Each SIG must have a charter that specifies its scope (topics, subsystems, code repos and directories), responsibilities, and areas of authority. Charters are submitted to the ONNX GitHub via PR for review and approval by the Steering Committee who will be looking to ensure the scope of the SIG as represented in the charter is reasonable. All SIGs are expected to follow the standards established by the Steering Committee for how Contributors are roles of authority/leadership are selected/granted, how decisions are made, and how conflicts are resolved.

A primary reason that SIGs exist is as forums for collaboration. Much work in a SIG should stay local within that SIG. However, SIGs must communicate in the open, ensure other SIGs and community members can find meeting notes, discussions, designs, and decisions, and periodically communicate a high-level summary of the SIG's work to the community. SIGs are also responsible to:

* Meet regularly, at least monthly
* Keep up-to-date meeting notes, linked from the SIG's page in the community repo
* Announce meeting agenda and minutes after each meeting, on their SIG mailing list and/or Slack channel
* Ensure the SIG's mailing list is archived (i.e on GitHub)
* Report activity in overall ONNX community meetings
* Participate in release planning meetings, retrospectives, etc (if relevant)
* Actively triage issues, PRs, test failures, etc. related to code and tests owned by the SIG
* Use the above forums as the primary means of working, communicating, and collaborating, as opposed to private emails and meetings

#### Decision making

When it is time to formalize the work-product from a SIG, votes are taken from every contributor who participates in the SIG. The list of active contributors is determined by the one (or two) SIG leads to ensure that only those who have actively participated in the SIG can vote. At this time there are no restrictions on how many contributors from any one Member Company can participate (and hence vote). The Steering Committee will monitor how the community behaves and apply constraints if needed in the future.

While most work shouldn’t require expensive coordination with other SIGs, there will be efforts (features, refactoring, etc.) that cross SIG boundaries. In this case, it is expected that the SIGs coordinate with each other and come to mutually agreed solutions. In some cases, it may make sense to form a Working Group for joint work. Cross-SIG coordination will naturally require more time and implies a certain amount of overhead. This is intentional to encourage changes to be well encapsulated whenever possible.

### WG - Working Groups

Working Groups (WGs) are primarily used to facilitate topics of discussion that cross SIG lines, or are topics which are short-lived and require a limited set of decisions to be agreed upon. Working groups:

* do not own code
* have a clear goal measured through specific deliverables
* will be disbanded after the goal is achieved

Working Groups can create specifications, recommendations, or implementations for submission to the relevant SIGs for approval and acceptance.

A list of all active, inactive, and completed working groups can be found in the [working-groups repository](https://github.com/onnx/working-groups)

Working Groups are formed by submitting a proposal via PR to the Steering Committee. The proposal should cover:

* what is the exact problem being worked on
* what are the exit criteria
* who are the chairs (up to 2)
* what are the meeting and discussion mechanics

Working Groups are disbanded when there is no activity for more than *3 months* or when the chair informs the Steering Committee.

## Repository Guidelines




The current guidelines for all repos under ONNX github.org could be found [here](repo_guidelines.md).

## CLA / DCO

As of October 2020, the CLA (https://cla-assistant.io/onnx/onnx) has been retired. All commits are subject to the DCO (https://www.developercertificate.com/) and need to be signed.
