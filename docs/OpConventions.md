# Operator Conventions

To maintain consistency in operator signatures, we use the following principles:
- All attribute names should be lower case and use underscores when it helps with readability
- Any input/output represented by a single letter is capitalized (i.e. X)
- Any input/output represented by a full word or multiple words is all lower case and uses underscores when it helps with readability
- Any input/output representing a bias tensor will utilize the name "B"
- Any input/output representing a weight tensor will utilize the name “W”
- “axes” is used when an input, output or attribute is representing multiple axes
- “axis” is used when an input, output or attribute is representing a single axis
- Default type for integers will use variants on int, no uints, default is int64
- Any instance of multiplying a weight tensor by input tensor and adding bias will following the format of W*X^T+B
