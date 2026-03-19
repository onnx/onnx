<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Steering Committee election guideline

## Introduction

To encourage community participation and wider adoption in the industry, ONNX has introduced [open governance](https://github.com/onnx/onnx/wiki/Expanded-ONNX-Steering-Committee-Announced!) in March 2018. The governance has three defined structures to propel the development of ONNX project forward: [Steering Committee](/community/readme.md#steering-committee), [Special Interest Groups (SIGs)](/community/readme.md#sig---special-interest-groups), and [Working Groups (WGs)](/community/readme.md#wg---working-groups). While SIGs and WGs primarily focus on the technical roadmap of ONNX, the Steering Committee is responsible for setting the vision and governance process of the ONNX community.

For the first year of its ONNX open governance, representatives from Facebook, Microsoft, AWS, Intel and Nvidia are chosen to serve as the ONNX Steering Committee to help guide the project. The Steering Committee will be elected by the [Contributors](/community/readme.md#community-roles) in its second year and will be re-elected every year.

This document is created to provide guidelines for the election process to ensure maximum transparency and fairness.


## Timeline

Candidate applications will be accepted in April, and the election will be held in May. The new term for Steering Committee begins on June 1st of the corresponding year. The following table outlines the schedule for the election process.

| Schedule     | Event               |
|:-------------|:--------------------|
| 1st Monday of April| Application for Steering Committee candidates open. |
| 3rd Monday of April| Candidates and their applications posted on github.|
| 1st Monday of May| Election begins.     |
| 2nd Monday of May| Election closes, and votes counted. Election results announced in the same week.|
| 3rd Monday of May| Previous Steering Committee to meet the newly elected Committee for official transition.|
| June 1 | New term begins with elected Steering Committee. Steering Committee Emeritus members help with the transition for the month of June.      |


## Eligibility

### Eligibility for Steering Committee candidacy
Candidates will be self-nominated, and they do not necessarily need to be a [Contributor](/community/readme.md#community-roles) to the ONNX project. The duties of the Steering Committee extend beyond simply contributing code to the ONNX project.


### Eligibility for voting

To participate in the Steering committee election, you must be a Contributor to the ONNX project. As defined in the community guideline, Contributor is sponsored by 2 approvers from different companies.

Contributors are further required to submit their github handle, email address, and affiliated company name to be eligible for voting. Any Contributor who has not submitted their information by before April 31st will not be able to participate in the election. The Steering Committee is currently reviewing options for collecting contributor information, and the best option will be notified to the Contributors shortly.

## Candidacy process

## Voting process

### General election procedure
In order to promote fairness, the Steering Committee has decided to limit 1 vote per Member Company. Contributors will be able to vote individually, but their votes will be rolled up to represent the vote of associated Member Company. This procedure will prevent large companies with lots of Contributors from dominating the election results.

### Voting mechanics and algorithm

The election will use [Condorcet ranking](https://en.wikipedia.org/wiki/Condorcet_method) with [Schulze method](https://en.wikipedia.org/wiki/Schulze_method). Condorcet ranking allows voters to indicate ranked preference for candidates, and Schultz method provides an algorithm to tally the overall preference.

For ONNX Steering Committee election, the Condorcet ranking with Schulze method will be performed twice. The individual Contributor votes gets tallied first to Member Companies, and the results of the Member Company votes are ranked again using the same method.

### Voting platform
We will use Condorcet Internet Voting Service ([civs.cs.cornell.edu](https://civs.cs.cornell.edu/)) to collect votes from Contributors.

After votes are casted, the results of individual votes will be uploaded to ONNX Github election directory to ensure transparency.

## Election officers and Steering Committee emeritus members

### Election officers
Two election officers will be chosen from the current Steering committee to oversee the election process. They are responsible for overseeing the progress of the election and ensure the process is correctly implemented. Their duties include coordinating election as shown in the timeline above, tallying votes and announcing results for the ONNX community.

### Steering Committee emeritus members
Two Steering Committee members will remain as emeritus members for the newly elected Committee to help with transition process for 1 month. If previous Steering Committee members are reelected, then they will guide the transition for the new members, and there will not be a separate Steering Committee emeritus members.



